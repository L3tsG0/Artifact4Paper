#!/usr/bin/env python3
# python3 src/place_stop_sign.py --host localhost --port 5555 --map Town04   --candidates stopline_candidates.json   --mesh-path "/Game/Carla/Static/TrafficSign/Stop/Stop_v01/SM_stopSign.SM_stopSign"   --lateral-offset 1.0 --forward-offset 0.0 --yaw-offset 90 --draw --z-offset 1.5
import argparse
import json
import sys
from typing import Dict, Optional

import carla


def _loc_from_dict(d: Dict[str, float]) -> carla.Location:
    return carla.Location(x=d["x"], y=d["y"], z=d["z"])


def _rot_from_dict(d: Dict[str, float]) -> carla.Rotation:
    return carla.Rotation(roll=d["roll"], pitch=d["pitch"], yaw=d["yaw"])


def _transform_from_dict(d: Dict[str, Dict[str, float]]) -> carla.Transform:
    return carla.Transform(_loc_from_dict(d["location"]), _rot_from_dict(d["rotation"]))


def _load_candidate(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("best", data)


def _score_stop_blueprint(bp_id: str) -> int:
    lower = bp_id.lower()
    score = 0
    if "stop" in lower:
        score += 2
    if "sign" in lower:
        score += 2
    if "traffic" in lower:
        score += 1
    if "prop" in lower:
        score += 1
    if "busstop" in lower or "bus_stop" in lower or "bus-stop" in lower:
        score -= 5
    return score


def _find_stop_blueprint(
    library: carla.BlueprintLibrary,
    blueprint_id: str,
    allow_busstop: bool,
) -> Optional[carla.ActorBlueprint]:
    if blueprint_id:
        bp = library.find(blueprint_id)
        return bp

    stop_bps = [bp for bp in library if "stop" in bp.id.lower()]
    if not stop_bps:
        return None

    scored = sorted(
        ((bp, _score_stop_blueprint(bp.id)) for bp in stop_bps),
        key=lambda item: item[1],
        reverse=True,
    )
    best_bp, best_score = scored[0]
    if best_score < 2:
        return None
    if not allow_busstop and "busstop" in best_bp.id.lower():
        return None
    return best_bp


def _set_mesh_path(
    blueprint: carla.ActorBlueprint,
    mesh_path: str,
) -> bool:
    if blueprint.has_attribute("mesh_path"):
        blueprint.set_attribute("mesh_path", mesh_path)
        return True
    if blueprint.has_attribute("mesh"):
        blueprint.set_attribute("mesh", mesh_path)
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Place a visible STOP sign near a stop line.")
    parser.add_argument("--host", default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=5555, help="CARLA port")
    parser.add_argument("--map", default="Town04", help="CARLA map name")
    parser.add_argument("--candidates", default="stopline_candidates.json", help="JSON path")
    parser.add_argument("--blueprint", default="", help="Blueprint id for STOP sign")
    parser.add_argument("--mesh-path", default="", help="UE asset path for static.prop.mesh")
    parser.add_argument("--mesh-scale", type=float, default=1.0, help="Scale for static.prop.mesh")
    parser.add_argument("--list", action="store_true", help="List stop-related blueprints and exit")
    parser.add_argument(
        "--allow-busstop",
        action="store_true",
        help="Allow bus stop blueprint when no stop sign is available",
    )
    parser.add_argument("--lateral-offset", type=float, default=1.0, help="Extra offset from lane edge (m)")
    parser.add_argument("--forward-offset", type=float, default=0.0, help="Offset along lane forward (m)")
    parser.add_argument("--z-offset", type=float, default=0.0, help="Extra vertical offset (m)")
    parser.add_argument(
        "--align-ground",
        action="store_true",
        help="Align mesh bottom to road surface after spawning",
    )
    parser.add_argument("--yaw-offset", type=float, default=180.0, help="Yaw offset for sign (deg)")
    parser.add_argument("--draw", action="store_true", help="Draw debug markers in the world")
    parser.add_argument("--draw-seconds", type=float, default=10.0, help="Debug draw duration")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    world = client.get_world()
    current_map = world.get_map().name
    if args.map and args.map not in current_map:
        world = client.load_world(args.map)
    world.wait_for_tick()

    library = world.get_blueprint_library()
    if args.list:
        stop_bps = sorted([bp.id for bp in library if "stop" in bp.id.lower()])
        if not stop_bps:
            print("No stop-related blueprints found.")
            return 1
        print("\n".join(stop_bps))
        return 0

    candidate = _load_candidate(args.candidates)
    stop_line_loc = _loc_from_dict(candidate["stop_line_location"])
    waypoint = world.get_map().get_waypoint(
        stop_line_loc, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if waypoint is None:
        print("Failed to get waypoint for stop line.")
        return 1

    forward = waypoint.transform.get_forward_vector()
    right = waypoint.transform.get_right_vector()
    lateral = (waypoint.lane_width * 0.5) + args.lateral_offset
    base_z = stop_line_loc.z if args.align_ground else stop_line_loc.z + args.z_offset
    target_loc = carla.Location(
        x=stop_line_loc.x + right.x * lateral + forward.x * args.forward_offset,
        y=stop_line_loc.y + right.y * lateral + forward.y * args.forward_offset,
        z=base_z,
    )
    target_rot = carla.Rotation(
        roll=0.0,
        pitch=0.0,
        yaw=waypoint.transform.rotation.yaw + args.yaw_offset,
    )
    sign_tf = carla.Transform(target_loc, target_rot)

    if args.mesh_path:
        blueprint = library.find("static.prop.mesh")
        if blueprint is None:
            print("static.prop.mesh is not available in this CARLA build.")
            return 1
        if not _set_mesh_path(blueprint, args.mesh_path):
            print("static.prop.mesh has no mesh_path/mesh attribute. Cannot set mesh.")
            return 1
        if args.mesh_scale != 1.0 and blueprint.has_attribute("scale"):
            blueprint.set_attribute("scale", str(args.mesh_scale))
    else:
        blueprint = _find_stop_blueprint(library, args.blueprint, args.allow_busstop)
    if blueprint is None:
        print("No suitable stop sign blueprint found. Run with --list or pass --blueprint.")
        if not args.allow_busstop:
            print("If only bus stop is available, pass --allow-busstop.")
        return 1

    actor = world.try_spawn_actor(blueprint, sign_tf)
    if actor is None:
        for dz in [0.3, 0.6, 1.0]:
            sign_tf.location.z = target_loc.z + dz
            actor = world.try_spawn_actor(blueprint, sign_tf)
            if actor is not None:
                break

    if actor is None:
        print("Failed to spawn stop sign.")
        return 1

    if args.align_ground:
        bbox = actor.bounding_box
        actor_tf = actor.get_transform()
        bbox_center = actor_tf.transform(bbox.location)
        bottom_z = bbox_center.z - bbox.extent.z
        ground_z = waypoint.transform.location.z
        dz = (ground_z - bottom_z) + args.z_offset
        new_loc = actor_tf.location
        new_loc.z += dz
        actor.set_transform(carla.Transform(new_loc, actor_tf.rotation))

    print(f"Spawned stop blueprint: id={actor.id} blueprint={blueprint.id}")
    print(f"Location: x={sign_tf.location.x:.3f} y={sign_tf.location.y:.3f} z={sign_tf.location.z:.3f}")
    print(f"Rotation: yaw={sign_tf.rotation.yaw:.2f}")

    if args.draw:
        debug = world.debug
        debug.draw_point(sign_tf.location, size=0.2, color=carla.Color(255, 0, 0), life_time=args.draw_seconds)
        debug.draw_point(stop_line_loc, size=0.2, color=carla.Color(0, 255, 0), life_time=args.draw_seconds)
        debug.draw_line(
            stop_line_loc,
            sign_tf.location,
            thickness=0.1,
            color=carla.Color(255, 255, 0),
            life_time=args.draw_seconds,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
