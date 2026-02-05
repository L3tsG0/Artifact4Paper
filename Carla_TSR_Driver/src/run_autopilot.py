#!/usr/bin/env python3
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# python3 src/run_autopilot.py --config configs/autopilot.json
import argparse
import csv
import json
import math
import random
import sys
import time
from collections import deque
from queue import Empty, Queue
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

def _ensure_carla_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    base = repo_root / "third_party" / "PythonAPI_full" / "carla"
    if not base.exists():
        return
    dist = base / "dist"
    egg = None
    if dist.exists():
        eggs = sorted(dist.glob("carla-*.egg"))
        if eggs:
            py_tag = f"py{sys.version_info.major}.{sys.version_info.minor}"
            for candidate in eggs:
                if py_tag in candidate.name:
                    egg = candidate
                    break
            if egg is None:
                for candidate in eggs:
                    if "py3" in candidate.name:
                        egg = candidate
                        break
            if egg is None:
                egg = eggs[0]
    base_str = str(base)
    if base_str not in sys.path:
        sys.path.insert(0, base_str)
    if egg:
        egg_str = str(egg)
        if egg_str not in sys.path:
            sys.path.insert(0, egg_str)


_ensure_carla_paths()

import carla

try:
    from carla.agents.navigation.controller import VehiclePIDController
except Exception:  # pragma: no cover - runtime dependency
    try:
        from agents.navigation.controller import VehiclePIDController
    except Exception as exc:  # pragma: no cover - runtime dependency
        print("Failed to import VehiclePIDController. Ensure CARLA PythonAPI is on PYTHONPATH.")
        print("Hint: add PythonAPI/carla and the .egg to PYTHONPATH, or remove pip-installed carla.")
        print(f"Import error: {exc}")
        sys.exit(1)


def _loc_from_dict(d: Dict[str, float]) -> carla.Location:
    return carla.Location(x=d["x"], y=d["y"], z=d["z"])


def _rot_from_dict(d: Dict[str, float]) -> carla.Rotation:
    return carla.Rotation(roll=d.get("roll", 0.0), pitch=d.get("pitch", 0.0), yaw=d["yaw"])


def _transform_from_dict(d: Dict[str, Dict[str, float]]) -> carla.Transform:
    return carla.Transform(_loc_from_dict(d["location"]), _rot_from_dict(d["rotation"]))


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(base: Path, maybe_path: str) -> Path:
    p = Path(maybe_path)
    if p.is_absolute():
        return p
    return (base.parent / p).resolve()


def _load_candidate(path: Path, key: str) -> Tuple[carla.Transform, carla.Location]:
    data = _load_json(path)
    if key and key in data:
        candidate = data[key]
    else:
        candidate = data.get("best", data)
    spawn_tf = _transform_from_dict(candidate["spawn_point"])
    stop_line_loc = _loc_from_dict(candidate["stop_line_location"])
    return spawn_tf, stop_line_loc


def _get_speed_kmh(vehicle: carla.Vehicle) -> float:
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def _spawn_camera(
    world: carla.World,
    vehicle: carla.Vehicle,
    camera_cfg: Dict[str, object],
) -> Tuple[carla.Sensor, Queue]:
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(int(camera_cfg.get("width", 1280))))
    camera_bp.set_attribute("image_size_y", str(int(camera_cfg.get("height", 720))))
    camera_bp.set_attribute("fov", str(float(camera_cfg.get("fov", 90.0))))

    tf_cfg = camera_cfg.get("transform", {})
    cam_tf = carla.Transform(
        carla.Location(
            x=float(tf_cfg.get("x", 1.5)),
            y=float(tf_cfg.get("y", 0.0)),
            z=float(tf_cfg.get("z", 1.4)),
        ),
        carla.Rotation(
            pitch=float(tf_cfg.get("pitch", 0.0)),
            yaw=float(tf_cfg.get("yaw", 0.0)),
        ),
    )
    camera = world.spawn_actor(camera_bp, cam_tf, attach_to=vehicle)
    queue = Queue()
    camera.listen(queue.put)
    return camera, queue


def _try_spawn_with_offsets(
    world: carla.World,
    blueprint: carla.ActorBlueprint,
    base_tf: carla.Transform,
    z_offsets: Tuple[float, ...],
) -> Optional[carla.Vehicle]:
    for dz in z_offsets:
        tf = carla.Transform(
            carla.Location(
                x=base_tf.location.x,
                y=base_tf.location.y,
                z=base_tf.location.z + dz,
            ),
            base_tf.rotation,
        )
        vehicle = world.try_spawn_actor(blueprint, tf)
        if vehicle is not None:
            return vehicle
    return None


def _spawn_vehicle_with_fallback(
    world: carla.World,
    blueprint: carla.ActorBlueprint,
    target_tf: carla.Transform,
) -> carla.Vehicle:
    vehicle = _try_spawn_with_offsets(world, blueprint, target_tf, (0.0, 0.5, 1.0, 1.5))
    if vehicle is not None:
        return vehicle

    spawn_points = world.get_map().get_spawn_points()
    for sp in spawn_points:
        vehicle = world.try_spawn_actor(blueprint, sp)
        if vehicle is not None:
            vehicle.set_simulate_physics(False)
            try:
                vehicle.set_transform(
                    carla.Transform(
                        carla.Location(
                            x=target_tf.location.x,
                            y=target_tf.location.y,
                            z=target_tf.location.z + 0.5,
                        ),
                        target_tf.rotation,
                    )
                )
            finally:
                vehicle.set_simulate_physics(True)
            return vehicle

    raise RuntimeError("Failed to spawn vehicle.")


class TemporalFilter:
    def __init__(self, buffer_size: int, trigger_threshold: int) -> None:
        self.buffer = deque(maxlen=buffer_size)
        self.trigger_threshold = trigger_threshold

    def update(self, detected: bool) -> bool:
        self.buffer.append(bool(detected))
        return sum(self.buffer) >= self.trigger_threshold


class DummyDetector:
    def __init__(self, mode: str, roi_min: float, roi_max: float) -> None:
        self.mode = mode
        self.roi_min = roi_min
        self.roi_max = roi_max

    def detect(self, dist_to_stop: float, image_path: Optional[Path]) -> Tuple[bool, float]:
        if self.mode == "always":
            return True, 1.0
        if self.mode == "roi":
            return (self.roi_min < dist_to_stop < self.roi_max), 1.0
        return False, 0.0


class ApiDetector:
    def __init__(self, url: str, classname: str, threshold: float, timeout_s: float) -> None:
        self.url = url
        self.classname = classname
        self.threshold = threshold
        self.timeout_s = timeout_s
        self.last_path: Optional[str] = None
        self.last_detected = False
        self.last_confidence = 0.0
        try:
            import requests  # type: ignore
        except Exception as exc:
            raise RuntimeError("requests is required for API detector") from exc
        self._requests = requests

    def _extract_confidence(self, payload: Union[dict, list, float, int]) -> float:
        if isinstance(payload, (float, int)):
            return float(payload)
        if isinstance(payload, list) and payload:
            head = payload[0]
            if isinstance(head, (float, int)):
                return float(head)
            if isinstance(head, dict):
                for key in ("confidence", "score", "prob", "probability"):
                    if key in head:
                        return float(head[key])
        if isinstance(payload, dict):
            for key in ("confidence", "score", "prob", "probability"):
                if key in payload:
                    return float(payload[key])
        return 0.0

    def detect(self, dist_to_stop: float, image_path: Optional[Path]) -> Tuple[bool, float]:
        if image_path is None:
            return False, 0.0
        path_str = str(image_path)
        if self.last_path == path_str:
            return self.last_detected, self.last_confidence
        payload = {"file": path_str, "classname": self.classname}
        try:
            resp = self._requests.post(self.url, json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()
            conf = self._extract_confidence(data)
            detected = conf >= self.threshold
        except Exception:
            detected = False
            conf = 0.0
        self.last_path = path_str
        self.last_detected = detected
        self.last_confidence = conf
        return detected, conf


class AsrCurve:
    def __init__(self, bins: Tuple[Tuple[float, float, float], ...]) -> None:
        if not bins:
            raise ValueError("ASR bins are empty.")
        self._bins = tuple(sorted(bins, key=lambda b: b[0]))

    @classmethod
    def from_jsonl(cls, path: Path, target: str, model: str, speed_kmh: float) -> "AsrCurve":
        record = None
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if (
                    str(data.get("target")) == target
                    and str(data.get("model")) == model
                    and float(data.get("speed_kmh", -1)) == float(speed_kmh)
                ):
                    record = data
                    break
        if record is None:
            raise ValueError(f"No ASR record found for target={target} model={model} speed_kmh={speed_kmh}")

        bins = []
        for key, value in record.get("asr_by_distance", {}).items():
            if "-" not in key:
                continue
            lo_str, hi_str = key.split("-", 1)
            bins.append((float(lo_str), float(hi_str), float(value)))
        return cls(tuple(bins))

    def get_asr(self, distance_m: float) -> float:
        if distance_m <= self._bins[0][0]:
            return self._bins[0][2]
        if distance_m >= self._bins[-1][1]:
            return self._bins[-1][2]
        for lo, hi, asr in self._bins:
            if lo <= distance_m < hi:
                return asr
        return self._bins[-1][2]


class HardDropInjector:
    def apply(self, detected: bool, asr_p: float, rng: random.Random) -> Tuple[bool, bool]:
        if detected and rng.random() < asr_p:
            return False, True
        return detected, False


class AttackDetector:
    def __init__(
        self,
        base_detector: Union[ApiDetector, DummyDetector],
        asr_curve: Optional[AsrCurve],
        rng: random.Random,
        enabled: bool,
    ) -> None:
        self.base_detector = base_detector
        self.asr_curve = asr_curve
        self.rng = rng
        self.enabled = enabled
        self.injector = HardDropInjector()

    def detect(self, dist_to_stop: float, image_path: Optional[Path]) -> Tuple[bool, bool, float, float, bool]:
        raw_detected, confidence = self.base_detector.detect(dist_to_stop, image_path)
        if not self.enabled or self.asr_curve is None:
            return raw_detected, raw_detected, confidence, 0.0, False
        asr_p = self.asr_curve.get_asr(dist_to_stop)
        detected, dropped = self.injector.apply(raw_detected, asr_p, self.rng)
        return raw_detected, detected, confidence, asr_p, dropped


class SysAdvAgent:
    def __init__(
        self,
        vehicle: carla.Vehicle,
        carla_map: carla.Map,
        stop_line_loc: carla.Location,
        config: Dict[str, object],
    ) -> None:
        control_cfg = config["control"]
        lateral_cfg = control_cfg["lateral"]
        long_cfg = control_cfg["longitudinal"]

        self.vehicle = vehicle
        self.carla_map = carla_map
        self.stop_line_loc = stop_line_loc
        self.lookahead_m = float(lateral_cfg["lookahead_m"])
        self.cruise_speed_kmh = float(control_cfg["cruise_speed_kmh"])
        self.stop_hold_distance_m = float(control_cfg["stop_hold_distance_m"])
        self.stop_hold_speed_kmh = float(control_cfg.get("stop_hold_speed_kmh", 0.5))
        self.stop_hold_requires_detection = bool(control_cfg.get("stop_hold_requires_detection", False))

        roi_cfg = config["roi"]
        self.roi_min = float(roi_cfg["min_m"])
        self.roi_max = float(roi_cfg["max_m"])
        self.brake_start_m = float(control_cfg.get("brake_start_m", self.roi_max))
        self.target_speed_k = float(control_cfg.get("target_speed_k", 0.0))

        tracking_cfg = config["tracking"]
        self.tracker = TemporalFilter(
            buffer_size=int(tracking_cfg["buffer_size"]),
            trigger_threshold=int(tracking_cfg["trigger_threshold"]),
        )

        detector_cfg = config.get("detector", {})
        detector_mode = detector_cfg.get("mode", "off")
        if detector_mode == "api":
            base_detector: Union[ApiDetector, DummyDetector] = ApiDetector(
                url=detector_cfg["url"],
                classname=detector_cfg.get("classname", "stop"),
                threshold=float(detector_cfg.get("confidence_threshold", 0.5)),
                timeout_s=float(detector_cfg.get("timeout_s", 2.0)),
            )
        else:
            base_detector = DummyDetector(
                mode=detector_mode,
                roi_min=self.roi_min,
                roi_max=self.roi_max,
            )

        attack_cfg = config.get("attack", {})
        attack_enabled = bool(attack_cfg.get("enabled", False))
        asr_curve = None
        if attack_enabled:
            config_base = Path(str(config.get("config_path", "")))
            if config_base.exists():
                asr_path = _resolve_path(config_base, attack_cfg["asr_path"])
            else:
                asr_path = Path(attack_cfg["asr_path"]).resolve()
            asr_curve = AsrCurve.from_jsonl(
                asr_path,
                target=str(attack_cfg.get("target", "STOP")),
                model=str(attack_cfg.get("model", "YOLOv5")),
                speed_kmh=float(attack_cfg.get("speed_kmh", 25)),
            )
        rng = random.Random(int(attack_cfg.get("seed", 0)))
        self.detector = AttackDetector(base_detector, asr_curve, rng, attack_enabled)

        if self.target_speed_k <= 0.0:
            self.target_speed_k = self.cruise_speed_kmh / max(1.0, self.roi_max)

        args_lateral = {
            "K_P": float(lateral_cfg["Kp"]),
            "K_I": float(lateral_cfg["Ki"]),
            "K_D": float(lateral_cfg["Kd"]),
        }
        args_long = {
            "K_P": float(long_cfg["Kp"]),
            "K_I": float(long_cfg["Ki"]),
            "K_D": float(long_cfg["Kd"]),
        }
        self.controller = VehiclePIDController(vehicle, args_lateral=args_lateral, args_longitudinal=args_long)

    def _get_target_waypoint(self) -> carla.Waypoint:
        loc = self.vehicle.get_location()
        wp = self.carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            raise RuntimeError("Failed to get waypoint for vehicle location.")
        next_wps = wp.next(self.lookahead_m)
        if not next_wps:
            return wp
        current_yaw = self.vehicle.get_transform().rotation.yaw
        best_wp = min(next_wps, key=lambda cand: abs(((cand.transform.rotation.yaw - current_yaw + 180.0) % 360.0) - 180.0))
        return best_wp

    def run_step(self, image_path: Optional[Path]) -> Tuple[carla.VehicleControl, Dict[str, float]]:
        dist_to_stop = self.vehicle.get_location().distance(self.stop_line_loc)
        current_speed = _get_speed_kmh(self.vehicle)
        raw_detected, detected, confidence, asr_p, attack_dropped = self.detector.detect(dist_to_stop, image_path)
        if not (self.roi_min < dist_to_stop < self.roi_max):
            raw_detected = False
            detected = False
            confidence = 0.0
            asr_p = 0.0
            attack_dropped = False
        confirmed = self.tracker.update(detected)
        stop_hold_allowed = (not self.stop_hold_requires_detection) or confirmed

        target_speed = self.cruise_speed_kmh
        if confirmed:
            if dist_to_stop <= self.brake_start_m:
                target_speed = max(0.0, dist_to_stop * self.target_speed_k)
        if dist_to_stop <= self.stop_hold_distance_m and stop_hold_allowed:
            target_speed = 0.0

        target_wp = self._get_target_waypoint()
        control = self.controller.run_step(target_speed, target_wp)
        hold = False
        if (
            dist_to_stop <= self.stop_hold_distance_m
            and stop_hold_allowed
            and current_speed <= self.stop_hold_speed_kmh
        ):
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = True
            hold = True

        meta = {
            "dist_to_stop_m": dist_to_stop,
            "target_speed_kmh": target_speed,
            "raw_detected": float(raw_detected),
            "detected": float(detected),
            "confirmed": float(confirmed),
            "confidence": float(confidence),
            "asr_p": float(asr_p),
            "attack_dropped": float(attack_dropped),
            "hold": float(hold),
        }
        return control, meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Skeleton CARLA TSR autopilot.")
    parser.add_argument("--config", default="configs/autopilot.json", help="Config JSON path")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _load_json(config_path)
    config["config_path"] = str(config_path)

    client = carla.Client(config["host"], int(config["port"]))
    client.set_timeout(10.0)

    world = client.get_world()
    current_map = world.get_map().name
    if config.get("map") and config["map"] not in current_map:
        world = client.load_world(config["map"])
    world.wait_for_tick()

    if "candidates_path" in config:
        candidates_path = _resolve_path(config_path, config["candidates_path"])
        spawn_tf, stop_line_loc = _load_candidate(candidates_path, config.get("candidate_key", "best"))
    else:
        spawn_tf = _transform_from_dict(config["spawn"])
        stop_line_loc = _loc_from_dict(config["stop_line"])

    vehicle_bp_id = config["vehicle"]["blueprint"]
    bp = world.get_blueprint_library().find(vehicle_bp_id)
    vehicle = _spawn_vehicle_with_fallback(world, bp, spawn_tf)

    original_settings = world.get_settings()
    sync_cfg = config.get("sync", {})
    settings_applied = False
    camera = None
    image_queue = None
    camera_out_dir = None
    latest_image_path: Optional[Path] = None
    log_writer = None
    log_file = None

    try:
        if sync_cfg.get("enabled", False):
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = float(sync_cfg.get("fixed_delta_seconds", 0.05))
            world.apply_settings(settings)
            settings_applied = True

        agent = SysAdvAgent(vehicle, world.get_map(), stop_line_loc, config)

        camera_cfg = config.get("camera", {})
        save_every = int(camera_cfg.get("save_every_n_frames", 0))
        if camera_cfg.get("enabled", False):
            camera, image_queue = _spawn_camera(world, vehicle, camera_cfg)
            camera_out_dir = _resolve_path(config_path, camera_cfg.get("out_dir", "logs/camera"))
            camera_out_dir.mkdir(parents=True, exist_ok=True)

        log_cfg = config.get("log", {})
        if log_cfg.get("enabled", False):
            log_path = _resolve_path(config_path, log_cfg.get("path", "logs/autopilot.csv"))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", newline="", encoding="utf-8")
            log_writer = csv.writer(log_file)
            log_writer.writerow(
                [
                    "time",
                    "speed_kmh",
                    "dist_to_stop_m",
                    "target_speed_kmh",
                    "raw_detected",
                    "detected",
                    "confirmed",
                    "confidence",
                    "asr_p",
                    "attack_dropped",
                    "hold",
                    "image_path",
                    "throttle",
                    "brake",
                    "steer",
                ]
            )

        start = time.time()
        runtime_cfg = config.get("runtime", {})
        max_seconds = float(runtime_cfg.get("max_seconds", 30.0))
        stop_end_speed_kmh = float(runtime_cfg.get("stop_end_speed_kmh", 0.0))
        stop_end_min_seconds = float(runtime_cfg.get("stop_end_min_seconds", 0.0))
        stop_end_require_hold = bool(runtime_cfg.get("stop_end_require_hold", False))
        stop_end_elapsed = 0.0
        camera_frame_count = 0
        last_tick_time = time.time()
        fixed_delta_seconds = float(sync_cfg.get("fixed_delta_seconds", 0.05))

        while True:
            if sync_cfg.get("enabled", False):
                frame_id = world.tick()
            else:
                frame_id = world.wait_for_tick().frame

            now = time.time()
            tick_dt = fixed_delta_seconds if sync_cfg.get("enabled", False) else (now - last_tick_time)
            last_tick_time = now

            if camera and image_queue and save_every > 0:
                try:
                    image = image_queue.get(timeout=1.0)
                    camera_frame_count += 1
                    if camera_frame_count % save_every == 0:
                        filename = (camera_out_dir / f"frame_{image.frame:06d}.png").resolve()
                        image.save_to_disk(str(filename))
                        latest_image_path = filename
                except Empty:
                    pass

            control, meta = agent.run_step(latest_image_path)
            vehicle.apply_control(control)
            speed_kmh = _get_speed_kmh(vehicle)

            if log_writer:
                log_writer.writerow(
                    [
                        time.time() - start,
                        speed_kmh,
                        meta["dist_to_stop_m"],
                        meta["target_speed_kmh"],
                        meta["raw_detected"],
                        meta["detected"],
                        meta["confirmed"],
                        meta["confidence"],
                        meta["asr_p"],
                        meta["attack_dropped"],
                        meta["hold"],
                        str(latest_image_path) if latest_image_path else "",
                        control.throttle,
                        control.brake,
                        control.steer,
                    ]
                )

            if stop_end_speed_kmh > 0.0:
                if speed_kmh <= stop_end_speed_kmh and (not stop_end_require_hold or meta["hold"] > 0.5):
                    stop_end_elapsed += tick_dt
                    if stop_end_elapsed >= stop_end_min_seconds:
                        break
                else:
                    stop_end_elapsed = 0.0

            if (time.time() - start) > max_seconds:
                break
    finally:
        vehicle.destroy()
        if camera:
            camera.stop()
            camera.destroy()
        if log_file:
            log_file.close()
        if settings_applied:
            world.apply_settings(original_settings)

    return 0


if __name__ == "__main__":
    sys.exit(main())
