#include <iostream>
#include <istream>
#include <ostream>
#include <boost/log/trivial.hpp>
#include <string>
#include "vector.h"

Vector::Vector() : m_data(nullptr), m_dim(0), m_owner(true) {}

Vector::~Vector()
{
    if (m_owner) delete [] m_data;
}

Vector::Vector(int dim, double value) : m_data(nullptr), m_dim(dim), m_owner(true) 
{
    m_data = new double[dim];
    for (int i = 0; i < dim; i++) {
        m_data[i] = value;
    }
}

Vector::Vector(int dim, double* data_ptr) : m_data(data_ptr), m_dim(dim), m_owner(false) 
{
}

Vector::Vector(const Vector& vec) : m_data(nullptr), m_dim(0), m_owner(true) 
{
    m_dim = vec.m_dim;
    m_data = new double[m_dim];
    std::copy(vec.m_data, vec.m_data + m_dim, m_data);
}

Vector::Vector(Vector&& vec) : m_data(nullptr), m_dim(0), m_owner(true) 
{
    m_dim = vec.m_dim;
    m_data = vec.m_data;
    m_owner = vec.m_owner;
    vec.m_dim = 0;
    vec.m_data = nullptr;
    vec.m_owner = true;
}

Vector& Vector::operator=(const Vector& vec)
{
    if (this != &vec) {
        if (m_dim != vec.dim()) {
            if (m_owner) delete [] m_data;
            m_dim = vec.m_dim;
            m_data = new double[m_dim];
            m_owner = true;
        } 
        std::copy(vec.m_data, vec.m_data + m_dim, m_data);
    }
    return *this;
}

Vector& Vector::operator=(Vector&& vec)
{
    if (this != &vec) {
        if (m_owner) delete [] m_data;
        m_dim = vec.m_dim;
        m_data = vec.m_data;
        m_owner = vec.m_owner;
        vec.m_dim = 0;
        vec.m_data = nullptr;
        vec.m_owner = true;
    }
    return *this;
}

// b = \alpha a + \beta b
void Vector::add_vec(double alpha, const Vector& a, double beta)
{
    BOOST_LOG_TRIVIAL(trace) << __PRETTY_FUNCTION__;
    if (m_dim != a.dim())
        throw std::string("Error");
    for (int i = 0; i < a.dim(); i++) {
        m_data[i] = alpha * a.data()[i] + beta * m_data[i];
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}

void Vector::add_row_sum_mat(double alpha, const Matrix& A, double beta)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    if (m_dim != A.num_cols())
        throw std::string("Error");
    for (int j = 0; j < A.num_cols(); j++) {
        m_data[j] *= beta;
        for (int i = 0; i < A.num_rows(); i++) {
            m_data[j] += A.data()[i*A.num_cols()+j];
        }
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}
