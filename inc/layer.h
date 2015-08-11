#pragma once

#include "matrix.h"
#include "vector.h"
#include "functor.h"

class Layer
{
public:
    Layer(int in_dim, int out_dim, Functor f = Sigmoid(), Functor diff_f = DiffSigmoid());
    Layer(const Layer& l);
    Layer(const Layer&& l);
    Layer& operator=(const Layer& l);
    Layer& operator=(const Layer&& l);
    ~Layer();
    void forward(const Matrix& in, Matrix& out);
    void backward(const Matrix& layer_in, const Matrix& nnet_out, const Matrix& labels, Matrix& out_deltas);
    void backward(const Matrix& layer_in, const Matrix& in_deltas, Matrix& out_deltas);
    void get_params(Matrix& W, Vector& b);
    void get_dims(int* in_dim, int* out_dim);
    void set_params(const Matrix& W, const Vector& b);
    void update_params(const Matrix& grad_W, const Vector& grad_b, double lrate);

private:
    int m_in_dim;
    int m_out_dim;
    Functor m_f;
    Functor m_diff_f;
    Matrix m_W;
    Vector m_b;
};
