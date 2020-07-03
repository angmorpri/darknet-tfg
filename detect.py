#!python3
#-*- coding: utf-8 -*-
"""
    Wrapper de Darknet para inferencia.

    Éste script ejecuta una detección básica de Darknet, tal y como ésta está
    implementada en el código fuente (detector.c), pero "envuelta" mediante
    *ctypes* en código Python que facilita su puesta en marcha.

    También está pensado para realizar la inferencia en múltiples imágenes,
    incluso obtener estadísticas y presentar gráficas.

    Basado en ./python/darknet.py. Se han tenido que añadir algunas funciones
    extra en C, para permitir algunas acciones que no podían ser envueltas
    mediante Python. Todas estas se hallan en 'src/py_utils.c'.

    Creado:                 09 May 2020
    Última modificación:    03 Jul 2020

    @author: Ángel Moreno Prieto

"""
import argparse
from ctypes import *
from datetime import datetime
import numpy
from os import listdir
from os.path import isdir, isfile, join
from pprint import pprint
import random

#
# Wrappers para variables y funciones en C.
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
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, \
                              POINTER(c_int), c_int, POINTER(c_int)]
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
# Clases definidas
#
class Accum (object):
    """Acumulador de estadísticas"""

    def __init__ (self):
        # Estadísticas a presentar
        self.total = 0.0    # Suma total
        self.mean = 0.0     # Media
        self.stdev = 0.0    # Desviación estándar
        self.max = 0.0      # Valor máximo
        self.min = 0.0      # Valor mínimo

        # Interno
        self._accum = list()

    def update (self, value):
        """Añade un nuevo valor"""
        self._accum.append(value)
        self.total = sum(self._accum)
        self.mean = self.total / len(self._accum)
        self.stdev = numpy.std(self._accum)
        self.max = max(self._accum)
        self.min = min(self._accum)

    def __str__ (self):
        ret = f"Mean: {self.mean}; StDev: {self.stdev};" \
              f"MaxVal: {self.max}; MinVal: {self.min}\n"
        return ret


class Detection (object):
    """Clase que adapta la estructura original "detección"

    Almacena información de una detección exacta en una inferencia. Un conjunto
    de objetos Detection conforman el resultado de una inferencia.

    En concreto, almacena:
        * classname (str): Nombre de la clase detectada
        * probability (float) ¡: Probabilidad de que sea la clase indicada
        * box (float * 4): Cuatro flotantes indicando la localización y tamaño
            exactos del recuadro que enmarca el objeto detectado
        * objectness (float): Confianza en que lo detectado sea un objeto.

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
        ret = f"'{self.classname}' with prob {self.prob:.4f} at "
        ret += f" x = {self.box_x:.4f};" \
               f" y = {self.box_y:.4f};" \
               f" width = {self.box_w:.4f};" \
               f" height = {self.box_h:.4f};" \
               f" and objectness = {self.objectness:.4f}\n"
        return ret


class YOLOResults (object):
    """Clase que almacena los resultados de una inferencia completa.

    Presenta resultados útiles sobre la ejecución.

    Atributos:
        * time (Accum): Almacena los tiempos de ejecución.
        * fps (float): Mantiene una relación de los FPS tras la última imagen
            ejecutada.
        * results (list): Lista de los resultados de cada una de las imágenes
            ejecutadas. Cada celda de la lista se corresponde a un diccionario
            con los siguientes elementos:
                - image_path (str): Path de la imagen.
                - time (float): Tiempo tardado en ejecutar.
                - current_fps (float): FPS en el momento de terminar esta imagen
                - detection (list): Lista de objetos Detection, con los resul-
                    tados de la inferencia en esta imagen.

    """
    def __init__ (self):
        self.time = Accum()
        self.fps = 0.0
        self.results = list()

    def append (self, img, time, dets, nboxes):
        """Añade nuevas imágenes a la lista de resultados."""
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
                    self.results[-1]['detection'].append(Detection(dets[box],
                                                                   clase))

    def get_fps (self):
        """Calcula los FPS actuales"""
        total_time = sum(res['time'] for res in self.results)
        self.fps = len(self.results) / total_time
        return self.fps

    def print (self):
        """Imprime por pantalla la ejecución total."""
        self.short_print()
        for result in self.results:
            print(f"Image: {result['image_path']}")
            print(f" - Time: {result['time']:.4f}")
            print(f" - Accum FPS: {result['current_fps']:.4f}")
            for det in result['detection']:
                print(" - ", det)
            print("\n######################################################\n")

    def short_print (self):
        print()
        print(f"Inference of {len(self.results)} images")
        print(f"Total time taken = {self.time.total:.4f}")
        print(f"Mean time per image = {self.time.mean:.4f}")
        print(f"Mean FPS = {self.fps:.4f}")
        print()


#
# Funciones
#
def detect (fdata, fcfg, fweight, fimages, thresh=.5, hier_thresh=.5, nms=.45,
            verbose=False):
    """Ejecuta un proceso de detección completo, de una o varias imágenes,
    en función de diferentes parámetros.

    Parámetros:
        * fdata (str): Archivo '.data' a utilizar.
        * fcfg (str): Archivo '.cfg' a utilizar.
        * fweight (str): Archivo '.weight' a utilizar.
        * fimages (str): Lista de paths a imágenes a utilizar.
        * thresh, hier_thresh (float): Mínimo límite a partir del cual se
            admiten las predicciones de la red. Por defecto, 0.50 ambos.
        * nms (float): Non-Maximum Supression, parámetro para eliminar detec-
            ciones redundantes. Por defecto, 0.45.
        * verbose (bool): Activar el modo verbose (por defecto desactivado).

    """
    meta = load_meta(fdata)

    # Cargar network
    net = load_network(fcfg, fweight, 0)
    set_batch_network(net, 1)
    srand(2222222)

    # Cargar NNPACK
    nnp_initialize()
    set_net_threadpool(net)

    # Ejecutando la inferencia para todas las imágenes pasadas
    if verbose: print(f"> Running for {len(fimages)} images...")
    results = YOLOResults()
    Detection.NCLASSES = meta.classes
    Detection.CLASS_NAMES = meta.names
    for image_path in fimages:
        # Cargando imagen
        if verbose: print(f"> Loading '{image_path}'...")
        img = load_image_thread(image_path, net)
        sized = letterbox_image_thread(img, net)

        # Prediciendo y detectando
        tstart = datetime.now()
        network_predict(net, sized.data)
        tstop = datetime.now()
        nboxes = c_int(0)
        nboxes_pointer = pointer(nboxes)
        dets = get_network_boxes(net, img.w, img.h, thresh, hier_thresh, None,
                                 1, nboxes_pointer)
        nboxes = nboxes_pointer[0]

        # Aplicando NMS
        if (nms):
            do_nms_sort(dets, nboxes, meta.classes, nms)

        # Obteniendo resultados
        results.append(image_path, tstop-tstart, dets, nboxes)

        # Liberando memoria reservada para la imagen
        free_detections(dets, nboxes)
        free_image(img)
        free_image(sized)

        if verbose: print("> Done.")

    # Liberando memoria general del programa
    free_net_threadpool(net)
    nnp_deinitialize()
    free_network(net)

    if verbose: print(f"> Finished in {results.time.total} seconds.")
    return results


#
# Main
#
if __name__ == "__main__":

    # Argumentos en línea de comandos:
    parser = argparse.ArgumentParser(description="Runs the YOLO Darknet")
    #   Relacionados con archivos
    parser.add_argument('-d', default="cfg/coco.data", dest="data", type=str,
                        help="Chooses the .data file")
    parser.add_argument('-c', default="cfg/yolov3-tiny.cfg", dest="cfg", type=str,
                        help="Chooes the .cfg file")
    parser.add_argument('-w', default="yolov3-tiny.weights", dest="weights", type=str,
                        help="Chooses the .weights file")
    parser.add_argument('-i', default="testing/dog.jpg", dest="images", type=str,
                        help="Chooses the image or images directory")

    #   Relacionados con hiperparámetros
    parser.add_argument('-t', '--thresh', default=.5, dest="thresh", type=float,
                        help="Changes the network detection threshold (default, 0.5)")
    parser.add_argument('-ht', '--hier-thresh', default=.5, dest="hthresh", type=float,
                        help="Changes the network detection hier threshold (default, 0.5)")
    parser.add_argument('--nms', default=.45, dest="nms", type=float,
                        help="Changes the Non-Maximum Supression value (default, 0.45)")

    #   Otros
    parser.add_argument('-v', '--verbose', action="store_true", dest="verbose",
                        help="Verbose mode")
    parser.add_argument('-n', default=-1, dest="limit", type=int,
                        help="Limits the number of images taken from given directory")
    parser.add_argument('--long-output', action="store_true", dest="lout",
                        help="Prints the long output on finish."\
                             "If only one image is given, this is the defualt option.")

    # Ejecución
    args = parser.parse_args()

    #print(args.images)
    #print(args.weights)

    # Convirtiendo imagen/carpeta en una lista de imágenes aleatoria.
    fimage = args.images
    images_list = list()
    if isdir(fimage):
        images_list = [join(fimage, file) \
                       for file in listdir(fimage) \
                       if isfile(join(fimage, file))]
        if args.limit > 0:
            random.shuffle(images_list)
            images_list = images_list[:args.limit]
    elif isfile(fimage):
        images_list = [fimage]
    images_list = [bytes(i, encoding="utf-8") for i in images_list]

    # Llamando al detector
    res = detect(bytes(args.data, encoding="utf-8"),
                 bytes(args.cfg, encoding="utf-8"),
                 bytes(args.weights, encoding="utf-8"),
                 images_list,
                 args.thresh, args.hthresh, args.nms, verbose=args.verbose)

    # Mostrando el resultado.
    if len(images_list) == 1:
        res.print()
    else:
        if args.lout:
            res.print()
        else:
            res.short_print()

