#include <iostream>
#include <fstream>
#include <boost/log/trivial.hpp>
#include <utility>
#include <string>
#include <json-c/json.h>
#include "nnet.h"
#include "activation.h"

NNet::~NNet()
{
    for (int i = 0; i < (int)m_layers.size(); i++) {
        delete m_layers[i];
    }
}

void NNet::init(const char* cfg_path)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    json_object* obj = json_object_from_file(cfg_path);

    json_object* value;
    if (!json_object_object_get_ex(obj, "nnet_params.learning_rate", &value))
        throw std::string("Error\n");
    m_learning_rate = json_object_get_double(value);

    if (!json_object_object_get_ex(obj, "nnet_params.num_iters", &value))
        throw std::string("Error\n");
    m_num_iters = json_object_get_int(value);

    if (!json_object_object_get_ex(obj, "nnet_params.activations", &value))
        throw std::string("Error\n");
    int layer_len = json_object_array_length(value);
    m_activations.resize(layer_len);
    for (int l = 0; l < layer_len; l++) {
        json_object* tmp = json_object_array_get_idx(value, l);
        std::string activation = json_object_get_string(tmp);
        for (int i = 0; i < ActivFunc::SIZE; i++) {
            if (!activation.compare(ActivFuncNames[i])) {
                m_activations[l] = i;
            }
        }
    }

    if (!json_object_object_get_ex(obj, "nnet_params.layers", &value))
        throw std::string("Error\n");
    int array_len = json_object_array_length(value);
    m_dims.resize(array_len);
    for (int i = 0; i < array_len; i++) {
        json_object* tmp = json_object_array_get_idx(value, i);
        m_dims[i] = json_object_get_int(tmp);
    }

    m_layers.resize(m_dims.size() - 1);
    if (m_layers.size() != m_activations.size())
        throw std::string("Error");
    for (int i = 0; i < (int)m_layers.size(); i++) {
        if (m_activations[i] == ActivFunc::IDENTITY) {
            m_layers[i] = create_layer(m_dims[i], m_dims[i+1], Identity(), DiffIdentity());
        }
        if (m_activations[i] == ActivFunc::SIGMOID) {
            m_layers[i] = create_layer(m_dims[i], m_dims[i+1], Sigmoid(), DiffSigmoid());
        }
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}

void NNet::forward(const Matrix& data)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    if (m_forward_buff.size() != m_dims.size() || 
           m_forward_buff[0].num_rows() != data.num_rows()) {
        m_forward_buff.resize(m_dims.size());
        for (int i = 0; i < (int)m_forward_buff.size(); i++) {
            if (i == 0) {
                m_forward_buff[i] = data;
                continue;
            }
            m_forward_buff[i] = std::move(Matrix(data.num_rows(), m_dims[i]));
        }
    } else {
        m_forward_buff[0] = data;
    }

    for (int i = 0; i < (int)m_forward_buff.size() - 1; i++) {
        m_layers[i]->forward(m_forward_buff[i], m_forward_buff[i+1]);
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}

void NNet::backward(const Matrix& labels)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    if (m_backward_buff.size() != m_dims.size() - 1 || 
           m_backward_buff.back().num_rows() != labels.num_rows()) {
        m_backward_buff.resize(m_dims.size() - 1);
        for (int i = 0; i < (int)m_backward_buff.size(); i++) {
            m_backward_buff[i] = std::move(Matrix(labels.num_rows(), m_dims[i+1]));
        }
    } else {
        m_backward_buff[m_backward_buff.size() - 1] = std::move(Matrix(labels.num_rows(), m_dims.back()));
    }

    int last_idx = m_forward_buff.size() - 1;
    m_layers.back()->first_backward(m_forward_buff[last_idx-1], m_forward_buff[last_idx], labels, m_backward_buff.back());

    for (int i = m_layers.size() - 2; i >= 0; i--) {
        m_layers[i]->backward(m_forward_buff[i], m_layers[i+1]->get_weigths(), m_backward_buff[i+1], m_backward_buff[i]);
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}

void NNet::get_dims(int size, int* dims)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    if (size != (int)m_dims.size()) throw std::string("Error");
    for (int i = 0; i < size; i++) {
        dims[i] = m_dims[i];
    }
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}

void NNet::get_layer_gradients(int index, Matrix& grad_W, Vector& grad_b)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    grad_W.add_matT_mat(1.0, m_forward_buff[index], m_backward_buff[index], 0.0);
    grad_b.add_row_sum_mat(1.0, m_backward_buff[index], 0.0);
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}

void NNet::update_layer_params(int index, const Matrix& grad_W, const Vector& grad_b)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    m_layers[index]->update_params(grad_W, grad_b, m_learning_rate);
    BOOST_LOG_TRIVIAL(trace) << "<< " << __PRETTY_FUNCTION__;
}

void NNet::save(const char* file_path) const
{
    std::ofstream file;
    file.open(file_path, std::ofstream::binary | std::ofstream::trunc);
    if (file.is_open()) {

        file << m_layers.size();

        for (int i = 0; i < (int)m_layers.size(); i++) {
            file << m_activations[i];

            Matrix W(m_dims[i], m_dims[i+1]);
            Vector b(m_dims[i+1]);

            m_layers[i]->get_params(W, b);

            file << W.num_rows() << W.num_cols();
            for (int n = 0; n < W.num_rows()*W.num_cols(); n++) {
                file << W.data()[n];
            }
            file << b.dim();
            for (int n = 0; n < b.dim(); n++) {
                file << b.data()[n];
            }
        }
        file.close();
    } else {
        throw std::string("Error");
    }
}

void NNet::read(const char* file_path)
{
    std::ifstream file;
    file.open(file_path, std::ofstream::binary);
    if (file.is_open()) {

        for (int i = 0; i < (int)m_layers.size(); i++) {
            delete m_layers[i];
        }

        int nnet_size;
        file >> nnet_size;
        m_layers.resize(nnet_size);
        m_activations.resize(nnet_size);
        m_dims.resize(nnet_size+1);

        for (int i = 0; i < nnet_size; i++) {
            file >> m_activations[i];

            int W_num_rows;
            int W_num_cols;

            file >> W_num_rows >> W_num_cols;
            Matrix W(W_num_rows, W_num_cols);
            for (int n = 0; n < W.num_rows()*W.num_cols(); n++) {
                file >> W.data()[n];
            }

            int b_dim;
            file >> b_dim;
            Vector b(b_dim);
            for (int n = 0; n < b.dim(); n++) {
                file >> b.data()[n];
            }

            m_dims[i] = W_num_rows;
            m_dims[i+1] = W_num_cols;

            if (m_activations[i] == ActivFunc::IDENTITY) {
                m_layers[i] = create_layer(m_dims[i], m_dims[i+1], Identity(), DiffIdentity());
            }
            if (m_activations[i] == ActivFunc::SIGMOID) {
                m_layers[i] = create_layer(m_dims[i], m_dims[i+1], Sigmoid(), DiffSigmoid());
            }
            m_layers[i]->set_params(W, b);
        }
        file.close();
    } else {
        throw std::string("Error");
    }
}
