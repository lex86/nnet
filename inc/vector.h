#pragma once

#include "matrix.h"

class Matrix;

class Vector
{
public:
    Vector();
    Vector(int dim);
    Vector(int dim, double* data_ptr);
    ~Vector();
    Vector(const Vector& vec);
    Vector(Vector&& vec);
    Vector& operator=(const Vector& vec);
    Vector& operator=(Vector&& vec);
    double* data() { return m_data; }
    const double* data() const { return m_data; }
    int dim() const { return m_dim; }
    void add_vec(double alpha, const Vector& a, double beta);
    void add_row_sum_mat(double alpha, const Matrix& A, double beta);

private:
    double* m_data;
    int m_dim;
    bool m_owner;
};
