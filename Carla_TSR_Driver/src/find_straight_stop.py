#!/usr/bin/env python3
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# python3 src/find_straight_stop.py --host localhost --port 5555 --map Town04 --distance 50 --yaw-threshold 3 --list --out stopline_candidates.json

import argparse
import json
import sys
from typing import Dict, List, Optional, Tuple

import carla


def _angle_diff_deg(a: float, b: float) -> float:
    diff = (a - b + 180.0) % 360.0 - 180.0
    return diff


def _loc_dict(loc: carla.Location) -> Dict[str, float]:
    return {"x": loc.x, "y": loc.y, "z": loc.z}


def _rot_dict(rot: carla.Rotation) -> Dict[str, float]:
    return {"roll": rot.roll, "pitch": rot.pitch, "yaw": rot.yaw}


def _transform_dict(tf: carla.Transform) -> Dict[str, Dict[str, float]]:
    return {"location": _loc_dict(tf.location), "rotation": _rot_dict(tf.rotation)}


def _pick_previous(
    current: carla.Waypoint,
    step: float,
    road_id: int,
    lane_id: int,
) -> Optional[carla.Waypoint]:
    prevs = current.previous(step)
    if not prevs:
        return None
    for wp in prevs:
        if wp.road_id == road_id and wp.lane_id == lane_id:
            return wp
    return None


def _trace_back(
    start_wp: carla.Waypoint,
    distance: float,
    step: float,
) -> Optional[List[carla.Waypoint]]:
    road_id = start_wp.road_id
    lane_id = start_wp.lane_id
    points = [start_wp]

    covered = 0.0
    current = start_wp
    while covered < distance:
        prev_wp = _pick_previous(current, step, road_id, lane_id)
        if prev_wp is None:
            return None
        points.append(prev_wp)
        current = prev_wp
        covered += step

    return points


def _check_straight(
    points: List[carla.Waypoint],
    yaw_threshold: float,
) -> Tuple[bool, float, float]:
    yaws = [wp.transform.rotation.yaw for wp in points]
    base_yaw = yaws[0]
    max_delta = max(abs(_angle_diff_deg(y, base_yaw)) for y in yaws)
    max_step_delta = 0.0
    for i in range(1, len(yaws)):
        step_delta = abs(_angle_diff_deg(yaws[i], yaws[i - 1]))
        if step_delta > max_step_delta:
            max_step_delta = step_delta

    if max_delta > yaw_threshold:
        return False, max_delta, max_step_delta

    for i, wp in enumerate(points):
        if i > 0 and wp.is_junction:
            return False, max_delta, max_step_delta
        if wp.lane_type != carla.LaneType.Driving:
            return False, max_delta, max_step_delta

    return True, max_delta, max_step_delta


def _get_stop_line_location(sign: carla.Actor) -> carla.Location:
    trigger = sign.trigger_volume
    return sign.get_transform().transform(trigger.location)


def _collect_stop_actors(world: carla.World) -> List[Dict[str, object]]:
    stops = []
    for sign in world.get_actors().filter("traffic.stop*"):
        stops.append(
            {
                "source": "actor",
                "id": sign.id,
                "name": "traffic.stop",
                "transform": sign.get_transform(),
                "stop_line_loc": _get_stop_line_location(sign),
            }
        )
    return stops


def _collect_stop_landmarks(carla_map: carla.Map) -> Tuple[List[Dict[str, object]], List[carla.Landmark]]:
    landmarks = carla_map.get_all_landmarks()
    stops = []
    for lm in landmarks:
        name = getattr(lm, "name", "") or ""
        lm_type = getattr(lm, "type", "") or ""
        if "stop" in name.lower() or "stop" in lm_type.lower():
            stops.append(
                {
                    "source": "landmark",
                    "id": lm.id,
                    "name": name,
                    "type": lm_type,
                    "transform": lm.transform,
                    "stop_line_loc": lm.transform.location,
                }
            )
    return stops, landmarks


def _collect_stop_env_objects(world: carla.World) -> List[Dict[str, object]]:
    stops = []
    for obj in world.get_environment_objects(carla.CityObjectLabel.TrafficSigns):
        name = getattr(obj, "name", "") or ""
        if "stop" not in name.lower():
            continue
        obj_id = getattr(obj, "id", None)
        if obj_id is None:
            obj_id = getattr(obj, "uid", -1)
        stops.append(
            {
                "source": "env",
                "id": obj_id,
                "name": name,
                "transform": obj.transform,
                "stop_line_loc": obj.transform.location,
            }
        )
    return stops


def main() -> int:
    parser = argparse.ArgumentParser(description="Find straight 50m approach to STOP sign.")
    parser.add_argument("--host", default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=5555, help="CARLA port")
    parser.add_argument("--map", default="Town01", help="CARLA map name")
    parser.add_argument("--distance", type=float, default=50.0, help="Approach distance in meters")
    parser.add_argument("--step", type=float, default=1.0, help="Trace step in meters")
    parser.add_argument(
        "--yaw-threshold",
        type=float,
        default=3.0,
        help="Max yaw deviation for straight segment (deg)",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional JSON output path for best candidate",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print all candidates",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "actor", "landmark", "env"],
        default="auto",
        help="Stop source: actor/landmark/env (auto prefers actor then landmark then env).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info about available stop sources.",
    )
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    world = client.get_world()
    current_map = world.get_map().name
    if args.map and args.map not in current_map:
        world = client.load_world(args.map)
    world.wait_for_tick()

    carla_map = world.get_map()
    stop_actors = _collect_stop_actors(world)
    stop_landmarks, all_landmarks = _collect_stop_landmarks(carla_map)
    stop_env = _collect_stop_env_objects(world)

    if args.debug:
        print(f"stop actors: {len(stop_actors)}")
        print(f"stop landmarks: {len(stop_landmarks)} / all landmarks: {len(all_landmarks)}")
        print(f"stop env objects: {len(stop_env)}")
        if not stop_landmarks and all_landmarks:
            names = sorted({getattr(lm, 'name', '') for lm in all_landmarks if getattr(lm, 'name', '')})
            print(f"landmark names (sample): {names[:20]}")
        if not stop_env:
            env_all = world.get_environment_objects(carla.CityObjectLabel.TrafficSigns)
            env_names = sorted({getattr(obj, 'name', '') for obj in env_all if getattr(obj, 'name', '')})
            print(f"env sign names (sample): {env_names[:20]}")

    if args.source == "actor":
        stop_sources = stop_actors
        source_label = "actor"
    elif args.source == "landmark":
        stop_sources = stop_landmarks
        source_label = "landmark"
    elif args.source == "env":
        stop_sources = stop_env
        source_label = "env"
    else:
        if stop_actors:
            stop_sources = stop_actors
            source_label = "actor"
        elif stop_landmarks:
            stop_sources = stop_landmarks
            source_label = "landmark"
        elif stop_env:
            stop_sources = stop_env
            source_label = "env"
        else:
            stop_sources = []
            source_label = "none"

    if not stop_sources:
        print("No stop sources found.")
        return 1
    print(f"Using stop source: {source_label}")

    candidates = []
    for src in stop_sources:
        stop_line_loc = src["stop_line_loc"]
        stop_wp = carla_map.get_waypoint(stop_line_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if stop_wp is None:
            continue

        points = _trace_back(stop_wp, args.distance, args.step)
        if points is None:
            continue

        ok, max_delta, max_step_delta = _check_straight(points, args.yaw_threshold)
        if not ok:
            continue

        spawn_wp = points[-1]
        candidate = {
            "stop_source": src["source"],
            "stop_source_id": src["id"],
            "stop_source_name": src.get("name", ""),
            "stop_source_type": src.get("type", ""),
            "stop_source_transform": _transform_dict(src["transform"]),
            "stop_line_location": _loc_dict(stop_line_loc),
            "stop_line_waypoint": _transform_dict(stop_wp.transform),
            "spawn_point": _transform_dict(spawn_wp.transform),
            "road_id": stop_wp.road_id,
            "lane_id": stop_wp.lane_id,
            "max_yaw_delta_deg": max_delta,
            "max_step_yaw_delta_deg": max_step_delta,
        }
        candidates.append(candidate)

    if not candidates:
        print("No straight stop-sign approaches found.")
        return 1

    candidates.sort(key=lambda x: (x["max_yaw_delta_deg"], x["max_step_yaw_delta_deg"]))
    best = candidates[0]

    print("Best candidate:")
    print(json.dumps(best, indent=2))

    if args.list:
        print("All candidates:")
        print(json.dumps(candidates, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"best": best, "candidates": candidates}, f, indent=2)
        print(f"Wrote: {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
