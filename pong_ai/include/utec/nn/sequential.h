#ifndef UTEC_NN_SEQUENTIAL_H
#define UTEC_NN_SEQUENTIAL_H

#include "layer.h"

using namespace utec::algebra;

namespace utec::neural_network
{

    template <typename T>
    class Sequential : public ILayer<T>
    {
    private:
        std::vector<std::unique_ptr<ILayer<T>>> layers;

    public:
        void add_layer(std::unique_ptr<ILayer<T>> layer)
        {
            layers.push_back(std::move(layer));
        }
        
        Tensor<T, 2> forward(const Tensor<T, 2> &x) override
        {
            Tensor<T, 2> output = x;
            for (auto &layer : layers)
            {
                output = layer->forward(output);
            }
            return output;
        }

        Tensor<T, 2> backward(const Tensor<T, 2> &grad) override
        {
            Tensor<T, 2> current_grad = grad;
            for (auto it = layers.rbegin(); it != layers.rend(); ++it)
            {
                current_grad = (*it)->backward(current_grad);
            }
            return current_grad;
        }

        void update(T lr) override
        {
            for (auto &layer : layers)
            {
                layer->update(lr);
            }
        }
    };

} // namespace utec::neural_network

#endif