#!/usr/bin/env python3
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# python3 src/set_night.py --host localhost --port 5555

import argparse
import sys
from pathlib import Path
from typing import Optional


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


def _set_if_attr(obj: object, name: str, value: Optional[float]) -> None:
    if value is None:
        return
    if hasattr(obj, name):
        setattr(obj, name, float(value))


def main() -> int:
    parser = argparse.ArgumentParser(description="Set CARLA weather to night-like settings.")
    parser.add_argument("--host", default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--timeout", type=float, default=5.0, help="Client timeout (seconds)")
    parser.add_argument("--sun-altitude", type=float, default=-20.0, help="Sun altitude angle (negative for night)")
    parser.add_argument("--sun-azimuth", type=float, default=0.0, help="Sun azimuth angle")
    parser.add_argument("--cloudiness", type=float, default=0.0, help="Cloudiness (0-100)")
    parser.add_argument("--precipitation", type=float, default=0.0, help="Precipitation (0-100)")
    parser.add_argument("--fog-density", type=float, default=0.0, help="Fog density (0-100)")
    parser.add_argument("--wetness", type=float, default=0.0, help="Wetness (0-100)")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    weather = world.get_weather()

    _set_if_attr(weather, "sun_altitude_angle", args.sun_altitude)
    _set_if_attr(weather, "sun_azimuth_angle", args.sun_azimuth)
    _set_if_attr(weather, "cloudiness", args.cloudiness)
    _set_if_attr(weather, "precipitation", args.precipitation)
    _set_if_attr(weather, "fog_density", args.fog_density)
    _set_if_attr(weather, "wetness", args.wetness)

    world.set_weather(weather)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
