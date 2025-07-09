#ifndef UTEC_NN_NEURAL_NETWORK_H
#define UTEC_NN_NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <stdexcept>
#include "layer.h"
#include "loss.h"
#include "../algebra/Tensor.h"

using namespace utec::algebra;

namespace utec::neural_network
{

    template <typename T>
    class NeuralNetwork
    {
    private:
        std::vector<std::unique_ptr<ILayer<T>>> layers;
        MSELoss<T> criterion;

        // Validate network architecture
        void validate_architecture() const
        {
            if (layers.empty())
            {
                throw std::logic_error("Neural network has no layers");
            }
        }

    public:
        void add_layer(std::unique_ptr<ILayer<T>> layer)
        {
            layers.push_back(std::move(layer));
        }

        Tensor<T, 2> forward(const Tensor<T, 2> &x)
        {
            validate_architecture();
            Tensor<T, 2> output = x;
            for (auto &layer : layers)
            {
                output = layer->forward(output);
            }
            return output;
        }

        void backward(const Tensor<T, 2> &grad)
        {
            validate_architecture();
            Tensor<T, 2> current_grad = grad;
            for (auto it = layers.rbegin(); it != layers.rend(); ++it)
            {
                current_grad = (*it)->backward(current_grad);
            }
        }

        void optimizer(T lr)
        {
            for (auto &layer : layers)
            {
                layer->update(lr);
            }
        }

        T train(const Tensor<T, 2> &X, const Tensor<T, 2> &Y, size_t epochs, T lr)
        {
            validate_architecture();
            T final_loss = 0;

            for (size_t epoch = 0; epoch < epochs; epoch++)
            {
                // Forward pass
                Tensor<T, 2> pred = forward(X);
                final_loss = criterion.forward(pred, Y);

                // Backward pass
                Tensor<T, 2> grad = criterion.backward();
                backward(grad);

                // Update weights
                optimizer(lr);
            }
            return final_loss;
        }
    };

} // namespace utec::neural_network

#endif // UTEC_NN_NEURAL_NETWORK_H