#ifndef UTEC_NN_NEURAL_NETWORK_H
#define UTEC_NN_NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <stdexcept>
#include "layer.h"
#include "loss.h"
#include "sequential.h" // Incluir Sequential
#include "../algebra/Tensor.h"

using namespace utec::algebra;

namespace utec::neural_network
{
    template <typename T>
    class NeuralNetwork
    {
    private:
        std::vector<std::unique_ptr<ILayer<T>>> layers; // Mantener estructura original
        MSELoss<T> criterion;

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

        // Nuevos métodos para manejo de parámetros
        size_t contar_parametros() const
        {
            size_t total = 0;
            for (const auto &layer : layers)
            {
                total += layer->contar_parametros();
            }
            return total;
        }

        std::vector<T> obtener_parametros() const
        {
            std::vector<T> params;
            for (const auto &layer : layers)
            {
                auto layer_params = layer->obtener_parametros();
                params.insert(params.end(), layer_params.begin(), layer_params.end());
            }
            return params;
        }

        void establecer_parametros(const std::vector<T> &new_params)
        {
            size_t start_index = 0;
            for (const auto &layer : layers)
            {
                size_t layer_param_count = layer->contar_parametros();
                if (layer_param_count == 0)
                    continue;

                std::vector<T> layer_params(
                    new_params.begin() + start_index,
                    new_params.begin() + start_index + layer_param_count);
                layer->establecer_parametros(layer_params);
                start_index += layer_param_count;
            }
        }
    };

} // namespace utec::neural_network

#endif // UTEC_NN_NEURAL_NETWORK_H