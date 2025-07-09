#ifndef UTEC_NN_LOSS_H
#define UTEC_NN_LOSS_H

#include "../algebra/Tensor.h"

using namespace utec::algebra;

namespace utec::neural_network
{

    template <typename T>
    class MSELoss
    {
    private:
        Tensor<T, 2> last_pred;
        Tensor<T, 2> last_target;

    public:
        T forward(const utec::algebra::Tensor<T, 2> &pred,
                  const utec::algebra::Tensor<T, 2> &target)
        {
            last_pred = pred;
            last_target = target;

            T loss = 0;
            size_t batch_size = pred.shape()[0];
            size_t features = pred.shape()[1];

            for (size_t i = 0; i < batch_size; i++)
            {
                for (size_t j = 0; j < features; j++)
                {
                    T diff = pred(i, j) - target(i, j);
                    loss += diff * diff;
                }
            }
            return loss / (batch_size * features);
        }

        Tensor<T, 2> backward()
        {
            size_t batch_size = last_pred.shape()[0];
            size_t features = last_pred.shape()[1];
            Tensor<T, 2> grad({batch_size, features});
            T scale = static_cast<T>(2) / (batch_size * features);

            for (size_t i = 0; i < batch_size; i++)
            {
                for (size_t j = 0; j < features; j++)
                {
                    grad(i, j) = scale * (last_pred(i, j) - last_target(i, j));
                }
            }
            return grad;
        }
    };

} // namespace utec::neural_network

#endif // UTEC_NN_LOSS_H