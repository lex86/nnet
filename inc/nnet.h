#pragma once

#include <vector>
#include <string>

#include "layer.h"
#include "vector.h"
#include "matrix.h"
#include "activation.h"

extern const char* ActivFuncNames[];

class NNet
{
public:
    NNet() {}
    NNet(const NNet& nnet) = delete;
    NNet(const NNet&& nnet) = delete;
    NNet& operator=(const NNet& nnet) = delete;
    NNet& operator=(const NNet&& nnet) = delete;
    ~NNet();
    void init(const char* cfg_path);
    void forward(const Matrix& data);
    void backward(const Matrix& labels);
    int get_num_iters() const { return m_num_iters; }
    int get_learning_rate() const { return m_learning_rate; }
    int size() const { return m_layers.size(); }
    void get_dims(int size, int* dims);
    LayerBase* get_layer(int index) { return m_layers[index]; }
    void get_layer_gradients(int index, Matrix& grad_W, Vector& grad_b);
    void update_layer_params(int index, const Matrix& grad_W, const Vector& grad_b);
    void save(const char* file_path) const;
    void read(const char* file_path);

private:
    std::vector<int> m_activations;
    std::vector<int> m_dims;
    std::vector<LayerBase*> m_layers;

    std::vector<Matrix> m_forward_buff;
    std::vector<Matrix> m_backward_buff;

    int m_num_iters;
    double m_learning_rate;
};
