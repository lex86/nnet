#pragma once

#include "matrix.h"

class Functor
{
public:
    virtual ~Functor() {}
    virtual Matrix operator()(const Matrix& W, const Matrix& X, const Vector& b);
};

class Sigmoid : public Functor
{
public:
    ~Sigmoid() {}
    Matrix operator()(const Matrix& W, const Matrix& X, const Vector& b);
};

class DiffSigmoid : public Functor
{
public:
    ~DiffSigmoid() {}
    Matrix operator()(const Matrix& W, const Matrix& X, const Vector& b);
};
