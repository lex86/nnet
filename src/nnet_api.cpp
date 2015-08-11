#include <iostream>
#include <boost/log/trivial.hpp>
#include "nnet_api.h"
#include "nnet.h"
#include "layer.h"
#include "matrix.h"
#include "vector.h"

extern "C" {

NNet* NNet_init(const char* cfg_path)
{
    try {
       BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
       NNet* obj = new NNet();
       obj->init(cfg_path);
       BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
       return obj;
    } catch (...) {
        std::cerr << "Error: NNet_init\n";
        return nullptr;
    }
}

void NNet_destroy(NNet* nnet)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    delete nnet;
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}

int NNet_forward(NNet* nnet, int num_rows, int num_cols, double* data)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    try {
        Matrix mat(num_rows, num_cols, data);
        nnet->forward(mat);
    } catch (...) {
        std::cerr << "Error: NNet_forward\n";
        return -1;
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
    return 0;
}

int NNet_backward(NNet* nnet, int num_rows, int num_cols, double* labels) 
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    try {
        Matrix mat(num_rows, num_cols, labels);
        nnet->backward(mat);
    } catch (...) {
        std::cerr << "Error: NNet_backward\n";
        return -1;
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
    return 0;
}

int NNet_size(NNet* nnet, int* size)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    try {
        *size = nnet->size();
    } catch (...) {
        std::cerr << "Error: NNet_size\n";
        return -1;
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
    return 0;
}

int NNet_get_layer_dims(NNet* nnet, int index, int* in_dim, int* out_dim)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    try {
        (nnet->get_layer(index)).get_dims(in_dim, out_dim);
    } catch (...) {
        std::cerr << "Error: NNet_get_layer_dims\n";
        return -1;
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
    return 0;
}
int NNet_get_layer_params(NNet* nnet, int index, int num_rows, int num_cols, double* weights, int dim, double* biases)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    try {
        Matrix W(num_rows, num_cols, weights);
        Vector b(dim, biases);
        (nnet->get_layer(index)).get_params(W, b);
    } catch (...) {
        std::cerr << "Error: NNet_get_layer_params\n";
        return -1;
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
    return 0;
}
int NNet_get_layer_gradients(NNet* nnet, int index, int num_rows, int num_cols, double* weight_gradients, int dim, double* bias_gradients)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    try {
        Matrix grad_W(num_rows, num_cols, weight_gradients);
        Vector grad_b(dim, bias_gradients);
        nnet->get_layer_gradients(index, grad_W, grad_b);
    } catch (...) {
        std::cerr << "Error: NNet_get_layer_gradients\n";
        return -1;
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
    return 0;
}
int NNet_set_layer_params(NNet* nnet, int index, int num_rows, int num_cols, double* weights, int dim, double* biases)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    try {
        Matrix W(num_rows, num_cols, weights);
        Vector b(dim, biases);
        (nnet->get_layer(index)).set_params(W, b);
    } catch (...) {
        std::cerr << "Error: NNet_set_layer_params\n";
        return -1;
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
    return 0;
}
int NNet_update_layer_params(NNet* nnet, int index, int num_rows, int num_cols, double* weight_gradients, int dim, double* bias_gradients)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    try {
        Matrix grad_W(num_rows, num_cols, weight_gradients);
        Vector grad_b(dim, bias_gradients);
        nnet->update_layer_params(index, grad_W, grad_b);
    } catch (...) {
        std::cerr << "Error: NNet_update_layer_params\n";
        return -1;
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
    return 0;
}

}

