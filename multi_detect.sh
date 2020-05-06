#!/bin/bash
# Multiple images detection in a row with standard Tiny YOLOv3 with
# pre-trained engine.
# Receives as a paremeter the folder path.

# Checking correct number of arguments
[[ $# -lt 1 ]] && { echo "Use: multiple_detect.sh <folder> [<limit>]"; exit 2; }

# If second argument is given, the number of images used will be cropped.
LIMIT=0
[[ $# -eq 2 ]] && { LIMIT="$2"; }

# Checking given image exists and running darknet
OUTFILE="images_array.txt"
RESULTS="multi_results.txt"
DIR="$1"
if [ -d "$DIR" ]; then
	ls -1d "$DIR"/* > "$OUTFILE"		# Each image path
	# Limiting, if necessary:
	if [[ $LIMIT -ne 0 ]]; then
		touch aux.txt
		head "-$LIMIT" "$OUTFILE" > aux.txt
		rm "$OUTFILE"
		mv aux.txt "$OUTFILE"
	fi

	# Running
	touch "$RESULTS"
	./darknet detector test cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights -dont_show < "$OUTFIE" > "$RESULTS"
else
	echo "$DIR does not exist."
fi
