#include <boost/log/trivial.hpp>
#include <string>
#include <algorithm>
#include <cblas.h>
#include <cmath>
#include "matrix.h"

Matrix::Matrix() : m_data(nullptr), m_num_rows(0), m_num_cols(0), m_owner(true) {} 

Matrix::~Matrix()
{
    if (m_owner) delete [] m_data;
}

Matrix::Matrix(int num_rows, int num_cols) : m_data(nullptr), m_num_rows(0), m_num_cols(0), m_owner(true) {
    m_num_rows = num_rows;
    m_num_cols = num_cols;
    m_data = new double[m_num_rows*m_num_cols];
}

Matrix::Matrix(int num_rows, int num_cols, double* data_ptr) : m_data(nullptr), m_num_rows(0), m_num_cols(0), m_owner(false)  {
    m_num_rows = num_rows;
    m_num_cols = num_cols;
    m_data = data_ptr;
}

Matrix::Matrix(const Matrix& mat) : m_data(nullptr), m_num_rows(0), m_num_cols(0), m_owner(true) 
{
    m_num_rows = mat.m_num_rows;
    m_num_cols = mat.m_num_cols;
    int num_elems = m_num_rows * m_num_cols;
    m_data = new double[num_elems];
    std::copy(mat.m_data, mat.m_data + num_elems, m_data);
}

Matrix::Matrix(Matrix&& mat) : m_data(nullptr), m_num_rows(0), m_num_cols(0), m_owner(true) 
{
    m_num_rows = mat.m_num_rows;
    m_num_cols = mat.m_num_cols;
    m_data = mat.m_data;
    m_owner = mat.m_owner;
    mat.m_num_rows = 0;
    mat.m_num_cols = 0;
    mat.m_data = nullptr;
}

Matrix& Matrix::operator=(const Matrix& mat)
{
    if (this != &mat) { 
        if (m_num_rows != mat.num_rows() || 
                m_num_cols != mat.num_cols()) {
            delete [] m_data;
            m_num_rows = mat.m_num_rows;
            m_num_cols = mat.m_num_cols;
            m_data = new double[m_num_rows * m_num_cols];
        }
        std::copy(mat.m_data, mat.m_data + m_num_rows * m_num_cols, m_data);
    }
    return *this;
}

Matrix& Matrix::operator=(Matrix&& mat)
{
    if (this != &mat) { 
        delete [] m_data;
        m_num_rows = mat.m_num_rows;
        m_num_cols = mat.m_num_cols;
        m_data = mat.m_data;
        mat.m_num_rows = 0;
        mat.m_num_cols = 0;
        mat.m_data = nullptr;
    }
    return *this;
}

// B = \alpha A + \beta B
void Matrix::add_mat(double alpha, const Matrix& A, double beta)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    // TODO: check dimensions
    int num_elems = m_num_rows*m_num_cols;
    for (int i = 0; i < num_elems; i++) {
        m_data[i] = alpha * A.data()[i] + beta * m_data[i];
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}
// B = \alpha A .* B
void Matrix::mul_mat(double alpha, const Matrix& A)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    // TODO: check dimensions
    int num_elems = m_num_rows*m_num_cols;
    for (int i = 0; i < num_elems; i++) {
        m_data[i] *= alpha* A.data()[i];
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}
// C = \alpha A^T B + \beta C
void Matrix::add_matT_mat(double alpha, const Matrix& A, const Matrix& B, double beta)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    if (m_num_rows != A.num_cols() ||
            m_num_cols != B.num_cols() ||
            A.num_rows() != B.num_rows()) {
        throw std::string("Error");
    }
    cblas_dgemm(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasTrans,
                 CBLAS_TRANSPOSE::CblasNoTrans, m_num_rows, m_num_cols,
                 A.num_rows(), alpha, A.data(),
                 A.num_cols(), B.data(), B.num_cols(),
                 beta, m_data, m_num_cols);
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}
// C = \alpha A B^T + \beta C
void Matrix::add_mat_matT(double alpha, const Matrix& A, const Matrix& B, double beta)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    if (m_num_rows != A.num_rows() ||
            m_num_cols != B.num_rows() ||
            A.num_cols() != B.num_cols()) {
        throw std::string("Error");
    }
    cblas_dgemm(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans,
                 CBLAS_TRANSPOSE::CblasTrans, m_num_rows, m_num_cols,
                 A.num_cols(), alpha, A.data(),
                 A.num_cols(), B.data(), B.num_cols(),
                 beta, m_data, m_num_cols);
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}
// C = \alpha A B + \beta C
void Matrix::add_mat_mat(double alpha, const Matrix& A, const Matrix& B, double beta)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    if (m_num_rows != A.num_rows() ||
            m_num_cols != B.num_cols() ||
            A.num_cols() != B.num_rows()) {
        throw std::string("Error");
    }
    cblas_dgemm(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans,
                 CBLAS_TRANSPOSE::CblasNoTrans, m_num_rows, m_num_cols,
                 A.num_cols(), alpha, A.data(),
                 A.num_cols(), B.data(), B.num_cols(),
                 beta, m_data, m_num_cols);
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}

// B[row] = \alpha v + \beta B[row] for all row
void Matrix::add_vec_to_rows(double alpha, const Vector& v, double beta)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    if (m_num_cols != v.dim()) 
        throw std::string("Error");

    for (int i = 0; i < m_num_rows; i++) {
        for (int j = 0; j < m_num_cols; j++) {
            m_data[i*m_num_cols+j] = alpha * v.data()[j] + beta * m_data[m_num_cols*i+j];
        }
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}
void Matrix::apply_sigmoid()
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    // TODO: limit double value
    int num_elems = m_num_rows * m_num_cols;
    for (int i = 0; i < num_elems; i++) {
        m_data[i] = 1.0L / (1.0L + exp(-m_data[i]));
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}
void Matrix::apply_diffsigmoid()
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    // TODO: limit double value
    int num_elems = m_num_rows * m_num_cols;
    for (int i = 0; i < num_elems; i++) {
        m_data[i] = 1.0L / (1.0L + exp(-m_data[i]));
        m_data[i] = m_data[i] * (1.0L + m_data[i]);
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}
