#include <boost/log/trivial.hpp>
#include "functor.h"

inline Matrix Functor::operator()(const Matrix& W, const Matrix& X, const Vector& b)
{
    BOOST_LOG_TRIVIAL(trace) << ">>Default " << __PRETTY_FUNCTION__;
    Matrix tmp(X.num_rows(), W.num_cols());
    tmp.add_vec_to_rows(1.0, b, 0.0);
    tmp.add_mat_mat(1.0, X, W, 1.0);
    BOOST_LOG_TRIVIAL(trace) << "<<Default " << __PRETTY_FUNCTION__;
    return tmp;
}

Matrix Sigmoid::operator()(const Matrix& W, const Matrix& X, const Vector& b)
{
    BOOST_LOG_TRIVIAL(trace) << ">>Sigmoid " << __PRETTY_FUNCTION__;
    Matrix tmp(X.num_rows(), W.num_cols());
    tmp.add_vec_to_rows(1.0, b, 0.0);
    tmp.add_mat_mat(1.0, X, W, 1.0);
    tmp.apply_sigmoid();
    BOOST_LOG_TRIVIAL(trace) << "<<Sigmoid " << __PRETTY_FUNCTION__;
    return tmp;
}

Matrix DiffSigmoid::operator()(const Matrix& W, const Matrix& X, const Vector& b)
{
    BOOST_LOG_TRIVIAL(trace) << ">>DiffSigmoid " << __PRETTY_FUNCTION__;
    Matrix tmp(X.num_rows(), W.num_cols());
    tmp.add_vec_to_rows(1.0, b, 0.0);
    BOOST_LOG_TRIVIAL(info) << X.num_rows() ;
    BOOST_LOG_TRIVIAL(info) << X.num_cols() ;
    BOOST_LOG_TRIVIAL(info) << W.num_rows() ;
    BOOST_LOG_TRIVIAL(info) << W.num_cols() ;
    tmp.add_mat_mat(1.0, X, W, 1.0);
    tmp.apply_diffsigmoid();
    BOOST_LOG_TRIVIAL(trace) << "<<DiffSigmoid " << __PRETTY_FUNCTION__;
    return tmp;
}
