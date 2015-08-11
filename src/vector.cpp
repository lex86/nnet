#include <boost/log/trivial.hpp>
#include "vector.h"

Vector::Vector() : m_data(nullptr), m_dim(0), m_owner(true) {}

Vector::~Vector()
{
    if (m_owner) delete [] m_data;
}

Vector::Vector(int dim) : m_data(nullptr), m_dim(0), m_owner(true) 
{
    m_dim = dim;
    m_data = new double[dim];
}

Vector::Vector(int dim, double* data_ptr) : m_data(nullptr), m_dim(0), m_owner(true) 
{
    m_dim = dim;
    m_data = data_ptr;
    m_owner = false;
}

Vector::Vector(const Vector& vec) : m_data(nullptr), m_dim(0), m_owner(true) 
{
    m_dim = vec.m_dim;
    m_data = new double[m_dim];
    for (int i = 0; i < m_dim; i++) {
        m_data[i] = vec.m_data[i];
    }
}

Vector::Vector(Vector&& vec) : m_data(nullptr), m_dim(0), m_owner(true) 
{
    m_dim = vec.m_dim;
    m_data = vec.m_data;
    vec.m_dim = 0;
    vec.m_data = nullptr;
}

Vector& Vector::operator=(const Vector& vec)
{
    if (this != &vec) {
        if (m_dim != vec.dim()) {
            delete [] m_data;
            m_dim = vec.m_dim;
            m_data = new double[m_dim];
        } 
        for (int i = 0; i < m_dim; i++) {
            m_data[i] = vec.m_data[i];
        }
    }
    return *this;
}

Vector& Vector::operator=(Vector&& vec)
{
    if (this != &vec) {
        delete [] m_data;
        m_dim = vec.m_dim;
        m_data = new double[m_dim];
        vec.m_dim = 0;
        vec.m_data = nullptr;
    }
    return *this;
}

// b = \alpha a + \beta b
void Vector::add_vec(double alpha, const Vector& a, double beta)
{
    BOOST_LOG_TRIVIAL(trace) << __PRETTY_FUNCTION__;
    // TODO: check dimensions
    for (int i = 0; i < a.dim(); i++) {
        m_data[i] = alpha * a.data()[i] + beta * m_data[i];
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}

void Vector::add_row_sum_mat(double alpha, const Matrix& A, double beta)
{
    BOOST_LOG_TRIVIAL(trace) << __PRETTY_FUNCTION__;
    // TODO: check dimensions
    for (int j = 0; j < A.num_rows(); j++) {
        m_data[j] *= beta;
        for (int i = 0; i < A.num_rows(); i++) {
            m_data[j] += A.data()[i*A.num_cols()+j];
        }
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}
