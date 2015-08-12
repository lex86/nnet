#include <boost/log/trivial.hpp>
#include "activation.h"

const char *ActivFuncNames[] = {
    "identity",
    "sigmoid",
    "softmax"
};


Matrix Identity::operator()(const Matrix& W, const Matrix& X, const Vector& b)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    Matrix tmp(X.num_rows(), W.num_cols());
    tmp.add_vec_to_rows(1.0, b, 0.0);
    tmp.add_mat_mat(1.0, X, W, 1.0);
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
    return tmp;
}

Matrix DiffIdentity::operator()(const Matrix& W, const Matrix& X, const Vector& b)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    Matrix tmp(X.num_rows(), W.num_cols(), 1.0L);
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
    return tmp;
}

Matrix Sigmoid::operator()(const Matrix& W, const Matrix& X, const Vector& b)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    Matrix tmp(X.num_rows(), W.num_cols());
    tmp.add_vec_to_rows(1.0, b, 0.0);
    tmp.add_mat_mat(1.0, X, W, 1.0);
    tmp.apply_sigmoid();
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
    return tmp;
}

Matrix DiffSigmoid::operator()(const Matrix& W, const Matrix& X, const Vector& b)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    Matrix tmp(X.num_rows(), W.num_cols());
    tmp.add_vec_to_rows(1.0, b, 0.0);
    tmp.add_mat_mat(1.0, X, W, 1.0);
    tmp.apply_diffsigmoid();
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
    return tmp;
}
