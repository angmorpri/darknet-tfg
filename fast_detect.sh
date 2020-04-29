#!/bin/bash
# Fast darknet detect using YOLOv3 configuration and weights
# Receives one parameter that must be an image from ./testing_images

# Checking correct number of arguments
[[ $# -ne 1 ]] && { echo "Use: fast_detect.sh <image>"; exit 2; }

# Checking given image exists and running darknet
IMAGE="testing_images/$1"
if [ -f "$IMAGE" ]; then
	identify -format "Image size: %wx%h\n\n" "$IMAGE"
	./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights "$IMAGE"
else
	echo "./testing_images/$IMAGE does not exist."
fi