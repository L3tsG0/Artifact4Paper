#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def _render_template(template: str, tag: str, index: int) -> str:
    return template.format(tag=tag, i=index)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_config(path: Path, config: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=True)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run autopilot multiple times with unique output dirs.")
    parser.add_argument(
        "--config",
        default="configs/autopilot_Classifier.json",
        help="Base config JSON path.",
    )
    parser.add_argument("--runs", type=int, default=4, help="Number of runs.")
    parser.add_argument("--start-index", type=int, default=1, help="Start index for naming.")
    parser.add_argument("--tag", default="batch_", help="Prefix tag for output names.")
    parser.add_argument(
        "--camera-dir-template",
        default="logs/camera/{tag}{i}",
        help="Camera output dir template with {tag} and {i}.",
    )
    parser.add_argument(
        "--log-path-template",
        default="logs/camera/{tag}{i}/autopilot.csv",
        help="Log path template with {tag} and {i}.",
    )
    parser.add_argument(
        "--keep-configs",
        action="store_true",
        help="Keep generated config files in configs/.",
    )
    args = parser.parse_args()

    base_config_path = Path(args.config)
    base_config = _load_config(base_config_path)
    configs_dir = base_config_path.parent

    for offset in range(args.runs):
        index = args.start_index + offset
        camera_out_dir = _render_template(args.camera_dir_template, args.tag, index)
        log_path = _render_template(args.log_path_template, args.tag, index)

        config = dict(base_config)
        camera_cfg = dict(config.get("camera", {}))
        camera_cfg["out_dir"] = camera_out_dir
        config["camera"] = camera_cfg

        log_cfg = dict(config.get("log", {}))
        log_cfg["path"] = log_path
        config["log"] = log_cfg

        temp_name = f".autopilot_batch_{args.tag}{index}.json"
        temp_config_path = configs_dir / temp_name
        _write_config(temp_config_path, config)

        try:
            subprocess.run(
                ["python3", "src/run_autopilot.py", "--config", str(temp_config_path)],
                check=True,
            )
        finally:
            if not args.keep_configs and temp_config_path.exists():
                temp_config_path.unlink()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
