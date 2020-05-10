/*
    Python Utils Header

    This module includes a bunch of functions that help making low-level
    operations later in the Python wrapper.

    Created:        09 May 20
    Last modified:  09 May 20
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

#endif
