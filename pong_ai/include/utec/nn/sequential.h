// sequential.h
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

        // Nuevas implementaciones requeridas
        size_t contar_parametros() const override
        {
            size_t total = 0;
            for (const auto &layer : layers)
            {
                total += layer->contar_parametros();
            }
            return total;
        }

        std::vector<T> obtener_parametros() const override
        {
            std::vector<T> params;
            for (const auto &layer : layers)
            {
                auto layer_params = layer->obtener_parametros();
                params.insert(params.end(), layer_params.begin(), layer_params.end());
            }
            return params;
        }

        void establecer_parametros(const std::vector<T> &new_params) override
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

#endif