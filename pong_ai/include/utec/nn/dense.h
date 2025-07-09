#ifndef UTEC_NN_DENSE_H
#define UTEC_NN_DENSE_H

#include "layer.h"
#include "../algebra/Tensor.h"
#include <cmath>
#include <cstdlib>

using namespace utec::algebra;

namespace utec::neural_network
{

    template <typename T>
    class Dense : public ILayer<T>
    {
    private:
        utec::algebra::Tensor<T, 2> W;      // Weights [in_feats, out_feats]
        utec::algebra::Tensor<T, 2> dW;     // Weight gradients
        utec::algebra::Tensor<T, 1> b;      // Biases [out_feats]
        utec::algebra::Tensor<T, 1> db;     // Bias gradients
        utec::algebra::Tensor<T, 2> last_x; // Last input cache

    public:
        Dense(size_t in_feats, size_t out_feats,
              const utec::algebra::Tensor<T, 2> &weights = utec::algebra::Tensor<T, 2>(),
              const utec::algebra::Tensor<T, 1> &biases = utec::algebra::Tensor<T, 1>())
        {
            if (weights.shape().empty())
            {
                // Initialize weights with He initialization (better for ReLU)
                W = utec::algebra::Tensor<T, 2>(in_feats, out_feats);
                T stddev = std::sqrt(2.0 / in_feats);
                for (size_t i = 0; i < in_feats; i++)
                {
                    for (size_t j = 0; j < out_feats; j++)
                    {
                        T val = static_cast<T>(rand()) / RAND_MAX;
                        W(i, j) = (val * 2 - 1) * stddev;
                    }
                }
            }
            else
            {
                W = weights;
            }

            if (biases.shape().empty())
            {
                b = utec::algebra::Tensor<T, 1>(out_feats);
                b.fill(0);
            }
            else
            {
                b = biases;
            }

            dW = utec::algebra::Tensor<T, 2>(W.shape());
            db = utec::algebra::Tensor<T, 1>(b.shape());
        }

        utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2> &x) override
        {
            // Verify input dimensions
            if (x.shape()[1] != W.shape()[0])
            {
                throw std::invalid_argument("Input features mismatch: expected " +
                                            std::to_string(W.shape()[0]) +
                                            ", got " + std::to_string(x.shape()[1]));
            }

            last_x = x;
            size_t batch_size = x.shape()[0];
            size_t out_features = b.shape()[0];
            utec::algebra::Tensor<T, 2> output(batch_size, out_features);

            for (size_t i = 0; i < batch_size; i++)
            {
                for (size_t j = 0; j < out_features; j++)
                {
                    T sum = 0;
                    for (size_t k = 0; k < x.shape()[1]; k++)
                    {
                        sum += x(i, k) * W(k, j);
                    }
                    output(i, j) = sum + b(j);
                }
            }
            return output;
        }

        utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2> &grad) override
        {
            // Compute gradients
            utec::algebra::Tensor<T, 2> xT = last_x.transpose_2d();
            dW = matmul(xT, grad);

            // Compute bias gradient
            db.fill(0);
            for (size_t i = 0; i < grad.shape()[0]; i++)
            {
                for (size_t j = 0; j < grad.shape()[1]; j++)
                {
                    db(j) += grad(i, j);
                }
            }

            // Compute input gradient
            utec::algebra::Tensor<T, 2> WT = W.transpose_2d();
            return matmul(grad, WT);
        }

        void update(T lr) override
        {
            // Update weights
            for (size_t i = 0; i < W.shape()[0]; i++)
            {
                for (size_t j = 0; j < W.shape()[1]; j++)
                {
                    W(i, j) -= lr * dW(i, j);
                }
            }

            // Update biases
            for (size_t j = 0; j < b.shape()[0]; j++)
            {
                b(j) -= lr * db(j);
            }
        }

    private:
        utec::algebra::Tensor<T, 2> matmul(const utec::algebra::Tensor<T, 2> &a,
                                           const utec::algebra::Tensor<T, 2> &b) const
        {
            if (a.shape()[1] != b.shape()[0])
            {
                throw std::invalid_argument("Matrix dimensions must agree for multiplication");
            }

            utec::algebra::Tensor<T, 2> result(a.shape()[0], b.shape()[1]);
            for (size_t i = 0; i < a.shape()[0]; i++)
            {
                for (size_t k = 0; k < a.shape()[1]; k++)
                {
                    T val = a(i, k);
                    for (size_t j = 0; j < b.shape()[1]; j++)
                    {
                        result(i, j) += val * b(k, j);
                    }
                }
            }
            return result;
        }
    };

} // namespace utec::neural_network

#endif // UTEC_NN_DENSE_H