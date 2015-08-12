#pragma once

#include "nnet.h"

extern "C"
{

NNet * NNet_init(const char* cfg_path);
void NNet_destroy(NNet* nnet);
int NNet_forward(NNet* nnet, int num_rows, int num_cols, double* data);
int NNet_backward(NNet* nnet, int num_rows, int num_cols, double* labels);
int NNet_size(NNet* nnet, int* size);
int NNet_num_iters(NNet* nnet, int* num_iters);
int NNet_get_dims(NNet* nnet, int num, int* dims);
int NNet_get_layer_params(NNet* nnet, int index, int num_rows, int num_cols, double* weights, int dim, double* biases);
int NNet_get_layer_gradients(NNet* nnet, int index, int num_rows, int num_cols, double* weight_gradients, int dim, double* bias_gradients);
int NNet_set_layer_params(NNet* nnet, int index, int num_rows, int num_cols, double* weights, int dim, double* biases);
int NNet_update_layer_params(NNet* nnet, int index, int num_rows, int num_cols, double* weight_gradients, int dim, double* bias_gradients);
    
}
