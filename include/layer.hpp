#pragma once

#include <random>

#include "blas/vector.hpp"
#include "blas/matrix.hpp"

namespace neural
{

double
sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

double
sigmid_derivative(double x)
{
    return x * (1.0 - x);
}

double
tanh(double x)
{
    return std::tanh(x);
}

double
tanh_derivative(double x)
{
    return 1.0 - x * x;
}
    
class layer
{
public:

    layer(size_t in_size, size_t out_size)
    :   _neurons(out_size),
        _weights(out_size, in_size),
        _bias(out_size)
    {
        randomize();
    }

    void
    randomize()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        for (auto& w : _weights)
            w = dis(gen);

        for (auto& b : _bias)
            b = dis(gen);
    }

    void
    activate()
    {
        // positive sigmoid
        for (auto& n : _neurons)
            n = tanh(n);
    }

    void
    feed_forward(const blas::vector<double>& input)
    {
        _neurons = _weights * input + _bias;
        activate();
    }

    void
    backpropagate(const blas::vector<double>& input, blas::vector<double>& error, double learning_rate)
    {
        // calculate error
        blas::vector<double> delta = _weights.transpose() * error;

        // calculate gradient
        blas::vector<double> gradient = _neurons;
        gradient = gradient * (1.0 - gradient);
        gradient = gradient * error;

        // update weights
        _weights = _weights + blas::vector<double>::outer_product(gradient, input) * learning_rate;

        // update bias
        _bias = _bias + gradient * learning_rate;

        // update error
        error = delta;
    }

    // getters

    const blas::vector<double>&
    neurons() const
    {
        return _neurons;
    }

private:

    blas::vector<double> _neurons;
    blas::matrix<double> _weights;
    blas::vector<double> _bias;
};

} // namespace neural
