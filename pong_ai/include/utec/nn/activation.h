#ifndef UTEC_NN_ACTIVATION_H
#define UTEC_NN_ACTIVATION_H

#include "layer.h"
#include "../algebra/Tensor.h"

using namespace utec::algebra;

namespace utec::neural_network
{

    template <typename T>
    class ReLU : public ILayer<T>
    {
    private:
        Tensor<T, 2> mask; // Mask for backpropagation

    public:
        Tensor<T, 2> forward(const Tensor<T, 2> &x) override
        {
            auto shape = x.shape();
            mask = Tensor<T, 2>(shape);
            Tensor<T, 2> output(shape);

            for (size_t i = 0; i < x.shape()[0]; i++)
            {
                for (size_t j = 0; j < x.shape()[1]; j++)
                {
                    T val = x(i, j);
                    if (val > 0)
                    {
                        output(i, j) = val;
                        mask(i, j) = 1;
                    }
                    else
                    {
                        output(i, j) = 0;
                        mask(i, j) = 0;
                    }
                }
            }
            return output;
        }

        Tensor<T, 2> backward(const Tensor<T, 2> &grad) override
        {
            Tensor<T, 2> output(grad.shape());
            for (size_t i = 0; i < grad.shape()[0]; i++)
            {
                for (size_t j = 0; j < grad.shape()[1]; j++)
                {
                    output(i, j) = grad(i, j) * mask(i, j);
                }
            }
            return output;
        }

        void update(T /*lr*/) override
        {
            // ReLU has no parameters to update
        }

        size_t contar_parametros() const override { return 0; }
        std::vector<T> obtener_parametros() const override { return {}; }
        void establecer_parametros(const std::vector<T> &) override {}
    };

} // namespace utec::neural_network

#endif // UTEC_NN_ACTIVATION_H