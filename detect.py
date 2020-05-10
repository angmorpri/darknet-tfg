"""
    Darknet Detect Python Wrapper

    This code runs Darknet's standard detect, copying the same system as in
    detector.c. It uses 'ctypes' in order to wrap some necessary C functions.

    It also allows, unlike original's detect, to bring a bunch of images at a
    time and calculate some parameters from it, such as FPS.

    The majority of this code style is copied from ./python/darknet.py, but
    improved.

    Also, some modifications needed to be done in the original C code, mainly,
    the ones involving NNPACK and Darknet Network manipulation. Those can be
    identified by the prefix 'py_' in the 'py_utils.c' source code.

    Created:        09 May 2020
    Last modified:  10 May 2020
"""

from __future__ import print_function
import argparse
from ctypes import *
from datetime import datetime
import numpy
from os import listdir
from os.path import isdir, isfile, join
from pprint import pprint
import random

#
# Wrapping
#
class BOX (Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION (Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

class IMAGE (Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA (Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

free_net_threadpool = lib.py_free_net_threadpool
free_net_threadpool.argtypes = [c_void_p]

free_network = lib.free_network
free_network.argtypes = [c_void_p]

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

letterbox_image_thread = lib.py_letterbox_image_thread
letterbox_image_thread.argtypes = [IMAGE, c_void_p]
letterbox_image_thread.restype = IMAGE

load_image_thread = lib.py_load_image_thread
load_image_thread.argtypes = [c_char_p, c_void_p]
load_image_thread.restype = IMAGE

load_meta = lib.get_metadata
load_meta.argtypes = [c_char_p]
load_meta.restype = METADATA

load_network = lib.load_network
load_network.argtypes = [c_char_p, c_char_p, c_int]
load_network.restype = c_void_p

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

nnp_deinitialize = lib.nnp_deinitialize

nnp_initialize = lib.nnp_initialize

set_net_threadpool = lib.py_set_net_threadpool
set_net_threadpool.argtypes = [c_void_p]

set_batch_network = lib.set_batch_network
set_batch_network.argtypes = [c_void_p, c_int]

srand = lib.srand
srand.argtypes = [c_int]


#
# Classes
#
class Accum (object):
    """Auxiliar class for statistical variables"""
    def __init__ (self):
        # Final stats
        self.total = 0.0
        self.mean = 0.0
        self.stdev = 0.0
        self.max = 0.0
        self.min = 0.0

        # Internal
        self.__accum = list()

    def update (self, value):
        """Adds a new value"""
        self.__accum.append(value)
        self.total = sum(self.__accum)
        self.mean = self.total / len(self.__accum)
        self.stdev = numpy.std(self.__accum)
        self.max = max(self.__accum)
        self.min = min(self.__accum)

    def __str__ (self):
        ret = "Mean: {}; StDev: {}; MaxVal: {}; MinVal: {}\n".format(self.mean, self.stdev, self.max, self.min)
        return ret


class Detection (object):
    """Detection object adaptation.
    Stores information of a unique detection: classname, probability, box,
    and objectness.
    """
    NCLASSES = None
    CLASS_NAMES = None

    def __init__ (self, det, iclass):
        self.classname = Detection.CLASS_NAMES[iclass]
        self.prob = det.prob[iclass]
        self.box_x = det.bbox.x
        self.box_y = det.bbox.y
        self.box_w = det.bbox.w
        self.box_h = det.bbox.h
        self.objectness = det.objectness

    def __str__ (self):
        ret = "Identified class '{}' with prob {:.4f} at ".format(self.classname, self.prob)
        ret += " x = {:.4f}; y = {:.4f}; width = {:.4f}; height = {:.4f}; and objectness = {:.4f}" \
            .format(self.box_x, self.box_y, self.box_w, self.box_h, self.objectness)
        return ret+'\n'


class YOLOResults (object):
    """This class stores the results of a YOLO execution.
    It presents useful information gatheredd from the execution.
    """
    def __init__ (self):
        # General stats
        self.time = Accum()
        self.fps = 0.0

        # Image results
        self.results = list()

    def append (self, img, time, dets, nboxes):
        """Appends a new result for an inference.
        They will be saved into *self.results* in a dictionary with format:
         + time
         + current_fps; which is the total FPS when this image passes
         + image_path
         + detections; which is a list of Detection objects
        """
        time = time.total_seconds()     # Casts to float
        self.time.update(time)
        self.results.append({'time': time,
                            'current_fps': 0.0,
                            'image_path': img,
                            'detection': list()})
        self.results[-1]['current_fps'] = self.get_fps()

        for box in range(nboxes):
            for clase in range(Detection.NCLASSES):
                if dets[box].prob[clase] > 0:
                    self.results[-1]['detection'].append(Detection(dets[box], clase))

    def get_fps (self):
        total_time = sum(res['time'] for res in self.results)
        self.fps = len(self.results) / total_time
        return self.fps

    def print (self):
        """Prints everything"""
        self.short_print()
        for result in self.results:
            print("Image: {}".format(result['image_path']))
            print(" - Time: {:.4f}".format(result['time']))
            print(" - Accum FPS: {:.4f}".format(result['current_fps']))
            for det in result['detection']:
                print(" - ", det)
            print("\n######################################################\n")

    def short_print (self):
        print()
        print("Inference of {} images".format(len(self.results)))
        print("Total time lasted = {:.4f}".format(self.time.total))
        print("Mean time per image = {:.4f}".format(self.time.mean))
        print("Mean FPS = {:.4f}".format(self.fps))
        print()


#
# Functions
#
def detect (fdata, fcfg, fweight, fimages, thresh=.5, hier_thresh=.5, nms=.45, verbose=False):
    """Main function.
    It receives the .data, .cfg and .weight files, and also, the path to a image
    or a folder with images, and makes the detection and returns some useful
    information.
    """
    meta = load_meta(fdata)

    # Loading network
    net = load_network(fcfg, fweight, 0)
    set_batch_network(net, 1)
    srand(2222222)

    # Loading NNPACK
    nnp_initialize()
    set_net_threadpool(net)

    # Loading image / images
    if verbose: print("> Running for {} images...".format(len(fimages)))

    # Running the inference iteration
    results = YOLOResults()
    Detection.NCLASSES = meta.classes
    Detection.CLASS_NAMES = meta.names
    for image_path in fimages:
        # Loading image
        if verbose: print("> Loading '{}'...".format(image_path))
        img = load_image_thread(image_path, net)
        sized = letterbox_image_thread(img, net)

        # Prediction and detection
        tstart = datetime.now()
        network_predict(net, sized.data)
        tstop = datetime.now()
        nboxes = c_int(0)
        nboxes_pointer = pointer(nboxes)
        dets = get_network_boxes(net, img.w, img.h, thresh, hier_thresh, None, 1, nboxes_pointer)
        nboxes = nboxes_pointer[0]

        # Applying Non-Maximum Supression
        if (nms):
            do_nms_sort(dets, nboxes, meta.classes, nms)

        # Gathering results
        results.append(image_path, tstop-tstart, dets, nboxes)

        # Freeing memory (per image)
        free_detections(dets, nboxes)
        free_image(img)
        free_image(sized)

        if verbose: print("> Done.")

    # Freeing memory (per execution)
    free_net_threadpool(net)
    nnp_deinitialize()

    free_network(net)

    if verbose: print("> Finished in {} seconds.".format(results.time.total))

    # Returning
    return results


#
# Main
#
if __name__ == "__main__":

    # Defaults
    fdata = "cfg/coco.data"
    fcfg = "cfg/yolov3-tiny.cfg"
    fweights = "yolov3-tiny.weights"
    fimage = "testing/"

    # Args parsing
    parser = argparse.ArgumentParser(description="Runs the YOLO Darknet")

    #   File-related arguments
    parser.add_argument('-d', default="cfg/coco.data", dest="data", type=str,
                        help="Chooses the .data file")
    parser.add_argument('-c', default="cfg/yolov3-tiny.cfg", dest="cfg", type=str,
                        help="Chooes the .cfg file")
    parser.add_argument('-w', default="yolov3-tiny.weights", dest="weights", type=str,
                        help="Chooses the .weights file")
    parser.add_argument('-i', default="data/dog.jpg", dest="images", type=str,
                        help="Chooses the image or images directory")

    #   Hyperparameter arguments
    parser.add_argument('-t', '--thresh', default=.5, dest="thresh", type=float,
                        help="Changes the network detection threshold (default, 0.5)")
    parser.add_argument('-ht', '--hier-thresh', default=.5, dest="hthresh", type=float,
                        help="Changes the network detection hier threshold (default, 0.5)")
    parser.add_argument('--nms', default=.45, dest="nms", type=float,
                        help="Changes the Non-Maximum Supression value (default, 0.45)")

    #   Others
    parser.add_argument('-v', '--verbose', action="store_true", dest="verbose",
                        help="Verbose mode")
    parser.add_argument('-n', default=-1, dest="limit", type=int,
                        help="Limits the number of images taken from given directory")
    parser.add_argument('--long-output', action="store_true", dest="lout",
                        help="Prints the long output on finish. If only one image is given, this is the defualt option.")

    #   Populating
    args = parser.parse_args()

    #   Taking images into a list
    print(args.images)
    print(args.weights)

    fimage = args.images
    images_list = list()
    if isdir(fimage):
        images_list = [join(fimage, file) for file in listdir(fimage) if isfile(join(fimage, file))]
        if args.limit > 0:
            random.shuffle(images_list)
            images_list = images_list[:args.limit]
    elif isfile(fimage):
        images_list = [fimage]

    # Calling the detector
    res = detect(args.data, args.cfg, args.weights, images_list,
                 args.thresh, args.hthresh, args.nms, verbose=args.verbose)

    # Presenting output
    if len(images_list) == 1:
        res.print()
    else:
        if args.lout:
            res.print()
        else:
            res.short_print()






