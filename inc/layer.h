#pragma once

#include "matrix.h"
#include "vector.h"
#include "activation.h"

class LayerBase
{
public:
    LayerBase() {}
    virtual ~LayerBase() {}
    virtual void forward(const Matrix& in, Matrix& out) = 0;
    virtual void first_backward(const Matrix& layer_in, const Matrix& nnet_out, const Matrix& labels, Matrix& layer_deltas) = 0;
    virtual void backward(const Matrix& layer_in, const Matrix& next_layer_W, const Matrix& next_layer_deltas, Matrix& layer_deltas) = 0;
    virtual const Matrix& get_weigths() const = 0;
    virtual void get_params(Matrix& W, Vector& b) = 0;
    //virtual void get_dims(int* in_dim, int* out_dim) = 0;
    virtual void set_params(const Matrix& W, const Vector& b) = 0;
    virtual void update_params(const Matrix& grad_W, const Vector& grad_b, double lrate) = 0;
};

template <typename Activation, typename DiffActivation>
class Layer : public LayerBase
{
public:
    Layer(int in_dim, int out_dim, Activation f, DiffActivation diff_f);
    Layer(const Layer& l);
    Layer(Layer&& l);
    Layer& operator=(const Layer& l);
    Layer& operator=(Layer&& l);
    ~Layer();
    void forward(const Matrix& in, Matrix& out);
    void first_backward(const Matrix& layer_in, const Matrix& nnet_out, const Matrix& labels, Matrix& layer_deltas);
    void backward(const Matrix& layer_in, const Matrix& next_layer_W, const Matrix& next_layer_deltas, Matrix& layer_deltas);
    virtual const Matrix& get_weigths() const;
    void get_params(Matrix& W, Vector& b);
    void set_params(const Matrix& W, const Vector& b);
    void update_params(const Matrix& grad_W, const Vector& grad_b, double lrate);

private:
    int m_in_dim;
    int m_out_dim;
    Activation m_f;
    DiffActivation m_diff_f;
    Matrix m_W;
    Vector m_b;
};

template <typename Activation, typename DiffActivation>
Layer<Activation, DiffActivation>* create_layer(int in, int out, Activation f, DiffActivation diff_f) {
    return new Layer<Activation, DiffActivation>(in, out, f, diff_f);
}

template <typename Activation, typename DiffActivation>
Layer<Activation, DiffActivation>::Layer(int in_dim, int out_dim, Activation f, DiffActivation diff_f) :
    m_in_dim(in_dim), m_out_dim(out_dim),
    m_f(f), m_diff_f(diff_f),
    m_W(Matrix(m_in_dim, m_out_dim)), m_b(Vector(m_out_dim))
{
}

template <typename Activation, typename DiffActivation>
Layer<Activation, DiffActivation>::~Layer()
{
}

template <typename Activation, typename DiffActivation>
Layer<Activation, DiffActivation>::Layer(const Layer<Activation, DiffActivation>& l) 
{
    m_in_dim = l.m_in_dim;
    m_in_dim = l.m_in_dim;
    m_f = l.m_f;
    m_diff_f = l.m_diff_f;
    m_W = l.m_W;
    m_b = l.m_b;
}

template <typename Activation, typename DiffActivation>
Layer<Activation, DiffActivation>::Layer(Layer<Activation, DiffActivation>&& l) 
{
    m_in_dim = l.m_in_dim;
    m_in_dim = l.m_in_dim;
    m_f = l.m_f;
    m_diff_f = l.m_diff_f;
    m_W = std::move(l.m_W);
    m_b = std::move(l.m_b);
}

template <typename Activation, typename DiffActivation>
Layer<Activation, DiffActivation>& Layer<Activation, DiffActivation>::operator=(const Layer& l) 
{
    if (this != &l) {
        m_in_dim = l.m_in_dim;
        m_in_dim = l.m_in_dim;
        m_f = l.m_f;
        m_diff_f = l.m_diff_f;
        m_W = l.m_W;
        m_b = l.m_b;
    }
    return *this;
}

template <typename Activation, typename DiffActivation>
Layer<Activation, DiffActivation>& Layer<Activation, DiffActivation>::operator=(Layer<Activation, DiffActivation>&& l) 
{
    if (this != &l) {
        m_in_dim = l.m_in_dim;
        m_in_dim = l.m_in_dim;
        m_f = l.m_f;
        m_diff_f = l.m_diff_f;
        m_W = std::move(l.m_W);
        m_b = std::move(l.m_b);
    }
    return *this;
}

template <typename Activation, typename DiffActivation>
void Layer<Activation, DiffActivation>::forward(const Matrix& in, Matrix& out)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    out = m_f(m_W, in, m_b);
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}

template <typename Activation, typename DiffActivation>
void Layer<Activation, DiffActivation>::first_backward(const Matrix& layer_in, const Matrix& nnet_out, const Matrix& labels, Matrix& layer_deltas)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    Matrix tmp;
    tmp = m_diff_f(m_W, layer_in, m_b);
    layer_deltas.add_mat(1.0, nnet_out, 0.0);
    layer_deltas.add_mat(-1.0, labels, 1.0);
    layer_deltas.mul_mat(1.0, tmp);
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}

template <typename Activation, typename DiffActivation>
void Layer<Activation, DiffActivation>::backward(const Matrix& layer_in, const Matrix& next_layer_W, const Matrix& next_layer_deltas, Matrix& layer_deltas)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    Matrix tmp;
    tmp = m_diff_f(m_W, layer_in, m_b);
    layer_deltas.add_mat_matT(1.0, next_layer_deltas, next_layer_W, 0.0);
    layer_deltas.mul_mat(1.0, tmp);
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}

template <typename Activation, typename DiffActivation>
const Matrix& Layer<Activation, DiffActivation>::get_weigths() const
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    return m_W;
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}

template <typename Activation, typename DiffActivation>
void Layer<Activation, DiffActivation>::get_params(Matrix& W, Vector& b)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    W = m_W;
    b = m_b;
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}
template <typename Activation, typename DiffActivation>
void Layer<Activation, DiffActivation>::set_params(const Matrix& W, const Vector& b)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    m_W = W;
    m_b = b;
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}
template <typename Activation, typename DiffActivation>
void Layer<Activation, DiffActivation>::update_params(const Matrix& grad_W, const Vector& grad_b, double lrate)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    m_W.add_mat(-lrate, grad_W, 1.0);
    m_b.add_vec(-lrate, grad_b, 1.0);
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}
