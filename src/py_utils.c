/*
    Funciones auxiliares en C para wrapper (cuerpo)

    Created:        09 May 20
    Last modified:  08 Nov 20
*/
#include "py_utils.h"

/* Definitions */
image py_load_image_thread (char *filename, network *net) {
    // Loads an image in a different thread.
    return load_image_thread(filename, 0, 0, net->c, net->threadpool);
}

image py_letterbox_image_thread (image img, network *net) {
    // Resizes an image in a different thread and returns.
    return letterbox_image_thread (img, net->w, net->h, net->threadpool);
}

void py_set_net_threadpool (network *net) {
    #ifdef QPU_GEMM
        net->threadpool = pthreadpool_create(1);
    #else
        net->threadpool = pthreadpool_create(4);
    #endif
}

void py_free_net_threadpool (network *net) {
    pthreadpool_destroy(net->threadpool);
}

void py_draw_predictions (image im, detection *dets, int nboxes, float thresh, char **names, int classes, const char *outfile) {
    image **alphabet = load_alphabet();
    draw_detections(im, dets, nboxes, thresh, names, alphabet, classes);
    save_image(im, outfile);
}
