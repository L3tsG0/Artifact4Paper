#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import carla


def _loc_from_dict(d: Dict[str, float]) -> carla.Location:
    return carla.Location(x=d["x"], y=d["y"], z=d["z"])


def _rot_from_dict(d: Dict[str, float]) -> carla.Rotation:
    return carla.Rotation(roll=d["roll"], pitch=d["pitch"], yaw=d["yaw"])


def _load_candidate(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("best", data)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _try_spawn_with_offsets(
    world: carla.World,
    blueprint: carla.ActorBlueprint,
    base_tf: carla.Transform,
    z_offsets: List[float],
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
) -> Tuple[Optional[carla.Vehicle], bool]:
    vehicle = _try_spawn_with_offsets(world, blueprint, target_tf, [0.0, 0.5, 1.0, 1.5])
    if vehicle is not None:
        return vehicle, False

    spawn_points = world.get_map().get_spawn_points()
    for sp in spawn_points:
        vehicle = world.try_spawn_actor(blueprint, sp)
        if vehicle is not None:
            try:
                vehicle.set_simulate_physics(False)
                fallback_tf = carla.Transform(
                    carla.Location(
                        x=target_tf.location.x,
                        y=target_tf.location.y,
                        z=target_tf.location.z + 0.5,
                    ),
                    target_tf.rotation,
                )
                vehicle.set_transform(fallback_tf)
            finally:
                vehicle.set_simulate_physics(True)
            return vehicle, True

    return None, False


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview STOP sign with a front camera.")
    parser.add_argument("--host", default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=5555, help="CARLA port")
    parser.add_argument("--map", default="Town04", help="CARLA map name")
    parser.add_argument("--candidates", default="stopline_candidates.json", help="JSON path")
    parser.add_argument("--outdir", default="outputs/preview", help="Output directory")
    parser.add_argument("--frames", type=int, default=5, help="How many frames to save")
    parser.add_argument("--width", type=int, default=1280, help="Image width")
    parser.add_argument("--height", type=int, default=720, help="Image height")
    parser.add_argument("--fov", type=float, default=90.0, help="Camera FOV")
    parser.add_argument("--cam-x", type=float, default=1.5, help="Camera X offset")
    parser.add_argument("--cam-y", type=float, default=0.0, help="Camera Y offset")
    parser.add_argument("--cam-z", type=float, default=1.4, help="Camera Z offset")
    parser.add_argument("--cam-pitch", type=float, default=0.0, help="Camera pitch")
    parser.add_argument("--cam-yaw", type=float, default=0.0, help="Camera yaw")
    parser.add_argument("--timeout", type=float, default=10.0, help="Max wait seconds")
    parser.add_argument("--draw", action="store_true", help="Draw debug markers in the world")
    parser.add_argument("--draw-seconds", type=float, default=5.0, help="Debug draw duration")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    world = client.get_world()
    current_map = world.get_map().name
    if args.map and args.map not in current_map:
        world = client.load_world(args.map)
    world.wait_for_tick()

    candidate = _load_candidate(args.candidates)
    sp = candidate["spawn_point"]
    spawn_loc = _loc_from_dict(sp["location"])
    spawn_rot = _rot_from_dict(sp["rotation"])
    spawn_tf = carla.Transform(spawn_loc, spawn_rot)
    stop_line_loc = _loc_from_dict(candidate["stop_line_location"])
    straight_dist = math.sqrt(
        (spawn_loc.x - stop_line_loc.x) ** 2
        + (spawn_loc.y - stop_line_loc.y) ** 2
        + (spawn_loc.z - stop_line_loc.z) ** 2
    )
    print(f"Spawn to stop line straight distance: {straight_dist:.2f} m")

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("vehicle.*model3*")
    if vehicle_bp:
        vehicle_bp = vehicle_bp[0]
    else:
        vehicle_bp = blueprint_library.filter("vehicle.*")[0]

    vehicle, used_fallback = _spawn_vehicle_with_fallback(world, vehicle_bp, spawn_tf)
    if vehicle is None:
        print("Failed to spawn vehicle at the candidate spawn point.")
        return 1
    if used_fallback:
        print("Spawned vehicle at a fallback location and teleported to target.")
    if args.draw:
        debug = world.debug
        debug.draw_point(spawn_loc, size=0.2, color=carla.Color(0, 255, 0), life_time=args.draw_seconds)
        debug.draw_point(stop_line_loc, size=0.2, color=carla.Color(255, 0, 0), life_time=args.draw_seconds)
        debug.draw_line(
            spawn_loc,
            stop_line_loc,
            thickness=0.1,
            color=carla.Color(255, 255, 0),
            life_time=args.draw_seconds,
        )
        debug.draw_string(
            stop_line_loc,
            f"STOP {straight_dist:.1f}m",
            color=carla.Color(255, 0, 0),
            life_time=args.draw_seconds,
        )

    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(args.width))
    camera_bp.set_attribute("image_size_y", str(args.height))
    camera_bp.set_attribute("fov", str(args.fov))

    cam_tf = carla.Transform(
        carla.Location(x=args.cam_x, y=args.cam_y, z=args.cam_z),
        carla.Rotation(pitch=args.cam_pitch, yaw=args.cam_yaw),
    )
    camera = world.spawn_actor(camera_bp, cam_tf, attach_to=vehicle)

    _ensure_dir(args.outdir)

    saved = {"count": 0}

    def _on_image(image: carla.Image) -> None:
        if saved["count"] >= args.frames:
            return
        filename = os.path.join(args.outdir, f"frame_{image.frame:06d}.png")
        image.save_to_disk(filename)
        saved["count"] += 1

    camera.listen(_on_image)

    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))

    start = time.time()
    try:
        while saved["count"] < args.frames and (time.time() - start) < args.timeout:
            world.wait_for_tick()
    finally:
        camera.stop()
        camera.destroy()
        vehicle.destroy()

    if saved["count"] == 0:
        print("No frames saved.")
        return 1

    print(f"Saved {saved['count']} frames to {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
