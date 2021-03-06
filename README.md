Based on <https://github.com/shizukachan/darknet-nnpack>

# Darknet with NNPACK for Raspberry Pi 3 B (Bachelor's Thesis Project)
## Abstract
This repository will contain all the workarounds of my BS thesis.
Mainly it will consists of Darknet implementation code and testing codes and data.
The goal is to measure and analyze the performance of YOLOv3 on a Raspberry.


## Build Instructions
This NNPACK implementation will make TinyYOLOv3 work with all the cores of the Raspberry, instead of the GPU.
### Components
* **NNPACK**, allows us to use all the CPU cores of the Raspberry.
* **Darknet**, YOLO CNN.
* **TinyYOLOv3**, pre-trained weights we will use for testing.
* [Optional] **ImageMagick**, to enhance some image manipulation and extra info.


### I. Required libraries
* **CMake** and **Ninja**, to build NNPACK.
* **Clang** (original [digitalbrain79](https://github.com/digitalbrain79/darknet-nnpack/) repo requests it, but it does not seem to be used anywhere).

		sudo apt-get install cmake clang
		git clone git://github.com/ninja-build/ninja.git
		cd ninja
		./configure.py --bootstrap
		sudo cp ninja /usr/sbin/
		export PATH="${PATH}:~/ninja"

Last line may be included in ~/.bashrc file, so it is loaded permanently. If you do so, you may also run `source ~/.bashrc` in order to upload the changes.


### II. NNPACK
	git clone https://github.com/shizukachan/NNPACK
	mkdir build
	cmake -G Ninja -D BUILD_SHARED_LIBS=ON -DCMAKE_C_FLAGS=-march=armv6k ..
	ninja
	sudo ninja install

Then you must create the file `/etc/ld.so.conf.d/nnpack.conf` and write `/usr/local/lib` in it, so that Darknet knows where NNPACK libraries are located.


### III. Darknet
In order to install Darknet, you may want just to clone this repository, which contains all the scritps for automatic inference and stats gathering:

	cd
	git clone https://github.com/angmorpri/darknet-tfg

On the other hand, you can just clone the original repository, in which this one is based:

	cd
	git clone https://github.com/shizukachan/darknet-nnpack

Then, you must `make` the repo. Be aware that the Makefile must have the flags NNPACK, NNPACK_FAST and ARM_NEON set to 1 in order to work properly.

	make
	sudo cp libdarknet.so /usr/local/lib/
	export LD_LIBRARY_PATH="/usr/local/lib:/usr/lib"

The last line can be included in ~/.bashrc file to load it permanently.


### IV. TinyYOLOv3 weights and first test
You may now download the TinyYOLOv3 weights. Those are optimized to work on constrained environments, like we are with the Raspberry. It is trained on COCO dataset.

	wget https://pjreddie.com/media/files/yolov3-tiny.weights

And then you can try it out:

	./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights testing/dog.jpg

If everything is fine, it should output something like:

	layer     filters    size              input                output
	    0 conv     16  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  16  0.150 BFLOPs
	    1 max          2 x 2 / 2   416 x 416 x  16   ->   208 x 208 x  16
	    2 conv     32  3 x 3 / 1   208 x 208 x  16   ->   208 x 208 x  32  0.399 BFLOPs
	    3 max          2 x 2 / 2   208 x 208 x  32   ->   104 x 104 x  32
	    4 conv     64  3 x 3 / 1   104 x 104 x  32   ->   104 x 104 x  64  0.399 BFLOPs
	    5 max          2 x 2 / 2   104 x 104 x  64   ->    52 x  52 x  64
	    6 conv    128  3 x 3 / 1    52 x  52 x  64   ->    52 x  52 x 128  0.399 BFLOPs
	    7 max          2 x 2 / 2    52 x  52 x 128   ->    26 x  26 x 128
	    8 conv    256  3 x 3 / 1    26 x  26 x 128   ->    26 x  26 x 256  0.399 BFLOPs
	    9 max          2 x 2 / 2    26 x  26 x 256   ->    13 x  13 x 256
	   10 conv    512  3 x 3 / 1    13 x  13 x 256   ->    13 x  13 x 512  0.399 BFLOPs
	   11 max          2 x 2 / 1    13 x  13 x 512   ->    13 x  13 x 512
	   12 conv   1024  3 x 3 / 1    13 x  13 x 512   ->    13 x  13 x1024  1.595 BFLOPs
	   13 conv    256  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 256  0.089 BFLOPs
	   14 conv    512  3 x 3 / 1    13 x  13 x 256   ->    13 x  13 x 512  0.399 BFLOPs
	   15 conv    255  1 x 1 / 1    13 x  13 x 512   ->    13 x  13 x 255  0.044 BFLOPs
	   16 detection
	   17 route  13
	   18 conv    128  1 x 1 / 1    13 x  13 x 256   ->    13 x  13 x 128  0.011 BFLOPs
	   19 upsample            2x    13 x  13 x 128   ->    26 x  26 x 128
	   20 route  19 8
	   21 conv    256  3 x 3 / 1    26 x  26 x 384   ->    26 x  26 x 256  1.196 BFLOPs
	   22 conv    255  1 x 1 / 1    26 x  26 x 256   ->    26 x  26 x 255  0.088 BFLOPs
	   23 detection
	Loading weights from yolov3-tiny.weights... (Version 2) Done!
	data/dog.jpg: Predicted in 5.103084 seconds.
	5
	Box 0 at (x,y)=(0.745051,0.209170) with (w,h)=(0.279108,0.171397)
	Box 1 at (x,y)=(0.509042,0.521773) with (w,h)=(0.480992,0.519662)
	Box 2 at (x,y)=(0.291488,0.621354) with (w,h)=(0.300040,0.581163)
	Box 3 at (x,y)=(0.324342,0.611325) with (w,h)=(0.312303,0.574243)
	Box 4 at (x,y)=(0.751867,0.218836) with (w,h)=(0.115131,0.108963)
	dog: 57%
	car: 52%
	truck: 56%
	car: 62%
	bicycle: 59%

More testing images can be found at `testing/`. You can also use `fast_detect.sh`, which runs the same darknet command but does not need to provide parameters, just the image name found at `testing/`.


## Testing scripts (WIP)
The following scripts are used to automatize some actions and gather stats. They are written in Python 3 (at least, Python 3.6). They also require `numpy` and `matplotlib` libraries, which can be installed like:

	sudo apt-get install libatlas-base-dev
	pip3 install numpy
	pip3 install matplotlib

### `detect.py`
(WIP)

### `plotter.py`
(WIP)

