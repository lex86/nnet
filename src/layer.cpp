#include <boost/log/trivial.hpp>
#include <utility>
#include "layer.h"

Layer::Layer(int in_dim, int out_dim, Functor f, Functor diff_f) :
    m_in_dim(in_dim), m_out_dim(out_dim),
    m_f(f), m_diff_f(diff_f),
    m_W(Matrix(m_in_dim, m_out_dim)), m_b(Vector(m_out_dim))
{
}

Layer::~Layer()
{
}

Layer::Layer(const Layer& l) 
{
    m_in_dim = l.m_in_dim;
    m_in_dim = l.m_in_dim;
    m_f = l.m_f;
    m_diff_f = l.m_diff_f;
    m_W = l.m_W;
    m_b = l.m_b;
}

Layer::Layer(const Layer&& l) 
{
    m_in_dim = l.m_in_dim;
    m_in_dim = l.m_in_dim;
    m_f = l.m_f;
    m_diff_f = l.m_diff_f;
    m_W = std::move(l.m_W);
    m_b = std::move(l.m_b);
}

Layer& Layer::operator=(const Layer& l) 
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

Layer& Layer::operator=(const Layer&& l) 
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

void Layer::forward(const Matrix& in, Matrix& out)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    out = m_f(m_W, in, m_b);
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}

void Layer::backward(const Matrix& layer_in, const Matrix& nnet_out, const Matrix& labels, Matrix& out_deltas)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    Matrix tmp;
    tmp = m_diff_f(m_W, layer_in, m_b);
    out_deltas.add_mat(1.0, nnet_out, 0.0);
    out_deltas.add_mat(-1.0, labels, 1.0);
    out_deltas.mul_mat(1.0, tmp);
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}

void Layer::backward(const Matrix& layer_in, const Matrix& in_deltas, Matrix& out_deltas)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    Matrix tmp;
    tmp = m_diff_f(m_W, layer_in, m_b);
    out_deltas.add_mat_matT(1.0, in_deltas, m_W, 0.0);
    out_deltas.mul_mat(1.0, tmp);
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}

void Layer::get_params(Matrix& W, Vector& b)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    W = m_W;
    b = m_b;
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}
void Layer::get_dims(int* in_dim, int* out_dim)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    *in_dim = m_in_dim;
    *out_dim = m_out_dim;
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}
void Layer::set_params(const Matrix& W, const Vector& b)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    m_W = W;
    m_b = b;
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}
void Layer::update_params(const Matrix& grad_W, const Vector& grad_b, double lrate)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    m_W.add_mat(-lrate, grad_W, 1.0);
    m_b.add_vec(-lrate, grad_b, 1.0);
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}
