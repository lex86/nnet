#pragma once

#include "vector.h"

class Vector;

class Matrix
{
public:
    Matrix(); 
    Matrix(int num_rows, int num_cols, double val = 0.0L);
    Matrix(int num_rows, int num_cols, double* data_ptr);
    Matrix(const Matrix& mat);
    Matrix(Matrix&& mat);
    ~Matrix();
    Matrix& operator=(const Matrix& mat);
    Matrix& operator=(Matrix&& mat);

    double* data() { return m_data; }
    const double* data() const { return m_data; }
    int num_rows() const { return m_num_rows; }
    int num_cols() const { return m_num_cols; }
    void add_mat(double alpha, const Matrix& A, double beta);
    void mul_mat(double alpha, const Matrix& A);
    void add_matT_mat(double alpha, const Matrix& A, const Matrix& B, double beta);
    void add_mat_matT(double alpha, const Matrix& A, const Matrix& B, double beta);
    void add_mat_mat(double alpha, const Matrix& A, const Matrix& B, double beta);
    void add_vec_to_rows(double alpha, const Vector& v, double beta);
    void apply_sigmoid();
    void apply_diffsigmoid();

private:
    double* m_data;
    int m_num_rows;
    int m_num_cols;
    bool m_owner;
};
