#pragma once

#include <vector>
#include <string>

#include "layer.h"
#include "vector.h"
#include "matrix.h"

class NNet
{
public:
    NNet() {}
    NNet(const NNet& nnet) = delete;
    NNet(const NNet&& nnet) = delete;
    NNet& operator=(const NNet& nnet) = delete;
    NNet& operator=(const NNet&& nnet) = delete;
    ~NNet() {}
    Layer& get_layer(int index) { return m_layers[index]; }
    void init(const char* cfg_path);
    void forward(const Matrix& data);
    void backward(const Matrix& labels);
    void get_layer_gradients(int index, Matrix& grad_W, Vector& grad_b);
    void update_layer_params(int index, const Matrix& grad_W, const Vector& grad_b);
    int get_num_iters() const { return m_num_iters; }
    int get_learning_rate() const { return m_learning_rate; }
    int size() const { return m_layers.size(); }

private:
    std::string m_activation;
    std::vector<int> m_dims;
    std::vector<Layer> m_layers;

    std::vector<Matrix> m_forward_buff;
    std::vector<Matrix> m_backward_buff;

    int m_num_iters;
    double m_learning_rate;
};
