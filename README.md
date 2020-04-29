Based on <https://github.com/zxzhaixiang/darknet-nnpack>

# Darknet with NNPACK for Raspberry Pi 3 B (End-of-degree Project)
## Abstract
This repository will contain all the workarounds of my TFG.
Mainly it will consists of Darknet implementation code and testing codes and data.
The goal is to measure and analyze the performance of YOLO on a Raspberry.

## Build Instructions
This implementation will make YOLO work with all the cores of the Raspberry, instead of the GPU.
### Components
* **NNPACK**, allows us to use all the CPU cores of the Raspberry.
* **Darknet**, YOLO CNN.
* **TinyYOLOv3**, pre-trained weights we will use for testing.

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
	cmake -G Ninja -D BUILD_SHARED_LIBS=ON ..
	ninja
	sudo ninja install

Then you must create the file `/etc/ld.so.conf.d/nnpack.conf` and write `/usr/local/lib` in it, so that Darknet knows where NNPACK libraries are located.

### III. Darknet
	cd
	git clone -b yolov3 https://github.com/zxzhaixiang/darknet-nnpack
	cd darknet-nnpack
	git checkout yolov3
	make
  
Be aware that the Makefile must have the flags NNPACK, NNPACK_FAST and ARM_NEON set to 1.

### IV. YOLOv3 weights and first test
You may now download the YOLOv3-Tiny weights. Those are optimized to work on constrained environments, like we are.

	wget https://pjreddie.com/media/files/yolov3-tiny.weights

And then you can try it out:

	./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights data/dog.jpg

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

You may also do then more testing with the other default images given at `data/`.


## Testing Experiments (WIP)
In order to run some of the tests described below, you will need the ImageMagick suite for editing images.

	sudo apt-get install imagemagick

### Measures to analyze (WIP)
* *Inference time*: Time taken in predict an image or a group of images.
* *Accuracy*
* *Billions FLOPS*
* *FPS*
