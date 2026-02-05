#!/usr/bin/env bash
# bash carla-server.sh [PORT]
PORT="${1:-2000}"

docker run \
  --gpus all \
  --net=host \
  -it \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --name carla-server \
  -v /home/tsuruoka/hdd/blender_file/Carla_TSR_Driver/assets:/home/carla/Import \
  carla:0.9.15-addassets \
  ./CarlaUE4.sh -RenderOffScreen -nosound -carla-port="$PORT" -quality-level=Low -opengl