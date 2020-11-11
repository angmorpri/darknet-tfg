/*
    Funciones auxiliares en C para wrapper (cabecera)

    Created:        09 May 20
    Last modified:  08 Nov 20
*/
#ifndef PYUTILS_H
#define PYUTILS_H

#include "darknet.h"
#include "image.h"

/* Declarations */
image py_load_image_thread (char *filename, network *net);
image py_letterbox_image_thread (image img, network *net);

void py_set_net_threadpool (network *net);
void py_free_net_threadpool (network *net);

void py_draw_predictions (image im, detection *dets, int nboxes, float thresh, char **names, int classes, const char *outfile);

#endif
