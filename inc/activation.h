#pragma once

#include "matrix.h"

enum ActivFunc {
    IDENTITY = 0,
    SIGMOID,
    SOFTMAX,
    SIZE
};


class Identity
{
public:
    Matrix operator()(const Matrix& W, const Matrix& X, const Vector& b);
};

class DiffIdentity
{
public:
    Matrix operator()(const Matrix& W, const Matrix& X, const Vector& b);
};


class Sigmoid
{
public:
    Matrix operator()(const Matrix& W, const Matrix& X, const Vector& b);
};

class DiffSigmoid
{
public:
    Matrix operator()(const Matrix& W, const Matrix& X, const Vector& b);
};
