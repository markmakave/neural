#pragma once

#include "blas/vector.hpp"
#include "blas/matrix.hpp"

#include "layer.hpp"

namespace neural
{

class network
{
    blas::vector<layer> _layers;

public:

    template <typename... Args>
    network(Args... args)
    :   _layers(sizeof...(Args) - 1)
    {
        static_assert(sizeof...(Args) >= 2, "network must have at least two layers");

        std::initializer_list<int> sizes = { args... };

        auto prev = sizes.begin();
        auto curr = prev + 1;
        for (auto& l : _layers)
            l = layer(*prev++, *curr++);
    }

    const blas::vector<double>&
    feed_forward(const blas::vector<double>& input)
    {
        _layers[0].feed_forward(input);

        for (size_t i = 1; i < _layers.size(); ++i)
        {
            _layers[i].feed_forward(_layers[i - 1].neurons());
        }

        return _layers.back().neurons();
    }

    void
    train(const blas::vector<double>& input, const blas::vector<double>& target, double learning_rate)
    {
        feed_forward(input);

        // calculate error
        blas::vector<double> error = _layers.back().neurons() - target;
        error = error * error;

        // backpropagate
        for (size_t i = _layers.size() - 1; i > 0; --i)
        {
            _layers[i].backpropagate(_layers[i - 1].neurons(), error, learning_rate);
        }
        _layers[0].backpropagate(input, error, learning_rate);
    }

};

}
