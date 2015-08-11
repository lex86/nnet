#include <boost/log/trivial.hpp>
#include <utility>
#include <json-c/json.h>
#include "nnet.h"

void NNet::init(const char* cfg_path)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    json_object* obj;
    obj = json_object_from_file(cfg_path);

    json_object* value;
    if (!json_object_object_get_ex(obj, "nnet_params.learning_rate", &value))
        throw std::string("Error\n");
    m_learning_rate = json_object_get_double(value);

    if (!json_object_object_get_ex(obj, "nnet_params.num_iters", &value))
        throw std::string("Error\n");
    m_num_iters = json_object_get_int(value);

    if (!json_object_object_get_ex(obj, "nnet_params.activation", &value))
        throw std::string("Error\n");
    m_activation = json_object_get_string(value);

    if (!json_object_object_get_ex(obj, "nnet_params.layers", &value))
        throw std::string("Error\n");
    int array_len = json_object_array_length(value);
    m_dims.resize(array_len);
    for (int i = 0; i < array_len; i++) {
        json_object* tmp = json_object_array_get_idx(value, i);
        m_dims[i] = json_object_get_int(tmp);
    }

    for (int i = 0; i < (int)m_dims.size(); i++) {
        if (m_activation == "sigmoid") {
            m_layers.push_back(std::move(Layer(m_dims[i], m_dims[i+1])));
        }
    }
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
        m_layers[i].forward(m_forward_buff[i], m_forward_buff[i+1]);
    }
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
    m_layers.back().backward(m_forward_buff[last_idx-1], m_forward_buff[last_idx], labels, m_backward_buff.back());

    for (int i = m_layers.size() - 1; i > 0; i--) {
        m_layers[i].backward(m_forward_buff[i-1], m_backward_buff[i], m_backward_buff[i-1]);
    }
}

void NNet::get_layer_gradients(int index, Matrix& grad_W, Vector& grad_b)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    grad_W.add_matT_mat(1.0, m_backward_buff[index], m_forward_buff[index], 0.0);
    grad_b.add_row_sum_mat(1.0, m_backward_buff[index], 0.0);
}

void NNet::update_layer_params(int index, const Matrix& grad_W, const Vector& grad_b)
{
    BOOST_LOG_TRIVIAL(trace) << ">> " << __PRETTY_FUNCTION__;
    m_layers[index].update_params(grad_W, grad_b, m_learning_rate);
}
