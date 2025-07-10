#ifndef UTEC_NN_DENSE_H
#define UTEC_NN_DENSE_H

#include "layer.h"
#include "../algebra/Tensor.h"
#include <cmath>
#include <cstdlib>
#include <iostream> // For debugging, can be removed later

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
        // Constructor for the Dense layer
        // in_feats: number of input features
        // out_feats: number of output features
        // weights: optional pre-initialized weights tensor. If default, weights are initialized.
        // biases: optional pre-initialized biases tensor. If default, biases are initialized to zeros.
        Dense(size_t in_feats, size_t out_feats,
              const utec::algebra::Tensor<T, 2> &weights = utec::algebra::Tensor<T, 2>(),
              const utec::algebra::Tensor<T, 1> &biases = utec::algebra::Tensor<T, 1>())
        {
            // Debugging output for constructor
            // std::cout << "Dense constructor called. in_feats: " << in_feats << ", out_feats: " << out_feats << "\n";
            // std::cout << "Provided weights shape: (" << weights.shape()[0] << ", " << weights.shape()[1] << ")\n";

            // Check if the provided weights tensor is effectively empty (dimensions are zero)
            // If it is, initialize weights with He initialization
            if (weights.shape()[0] == 0 || weights.shape()[1] == 0)
            {
                W = utec::algebra::Tensor<T, 2>(in_feats, out_feats);
                // He initialization for ReLU activation
                T stddev = std::sqrt(2.0 / in_feats);
                for (size_t i = 0; i < in_feats; i++)
                {
                    for (size_t j = 0; j < out_feats; j++)
                    {
                        // Generate random float between -1 and 1
                        T val = static_cast<T>(rand()) / RAND_MAX; // val is [0, 1]
                        W(i, j) = (val * 2 - 1) * stddev;          // Scale to [-stddev, stddev]
                    }
                }
                // std::cout << "Initialized W with shape: (" << W.shape()[0] << ", " << W.shape()[1] << ")\n";
            }
            else
            {
                // Use the provided weights
                W = weights;
                // std::cout << "Using provided W with shape: (" << W.shape()[0] << ", " << W.shape()[1] << ")\n";
            }

            // Check if the provided biases tensor is effectively empty (dimension is zero)
            // If it is, initialize biases to zeros
            if (biases.shape()[0] == 0)
            {
                b = utec::algebra::Tensor<T, 1>(out_feats);
                b.fill(0); // Initialize biases to zero
                // std::cout << "Initialized b with shape: (" << b.shape()[0] << ")\n";
            }
            else
            {
                // Use the provided biases
                b = biases;
                // std::cout << "Using provided b with shape: (" << b.shape()[0] << ")\n";
            }

            // Initialize gradients for weights and biases with the same shape
            dW = utec::algebra::Tensor<T, 2>(W.shape());
            db = utec::algebra::Tensor<T, 1>(b.shape());

            // std::cout << "Dense constructor finished. Current W shape: (" << W.shape()[0] << ", " << W.shape()[1] << ")\n";
            // std::cout << "Current b shape: (" << b.shape()[0] << ")\n";
        }

        // Performs the forward pass of the dense layer: output = input * W + b
        // x: input tensor (batch_size, in_features)
        // Returns: output tensor (batch_size, out_features)
        utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2> &x) override
        {
            // Debugging output for forward pass
            // std::cout << "Dense forward: Input x shape: (" << x.shape()[0] << ", " << x.shape()[1] << ")\n";
            // std::cout << "Dense forward: Layer W shape: (" << W.shape()[0] << ", " << W.shape()[1] << ")\n";

            // Store the input for use in the backward pass
            last_x = x;

            // Verify input dimensions match the weights
            if (x.shape()[1] != W.shape()[0])
            {
                throw std::invalid_argument("Input features mismatch: expected " +
                                            std::to_string(W.shape()[0]) +
                                            ", got " + std::to_string(x.shape()[1]));
            }

            // Perform matrix multiplication: output = x * W
            utec::algebra::Tensor<T, 2> output = matmul(x, W);

            // Add bias to each row of the output
            for (size_t i = 0; i < output.shape()[0]; i++)
            {
                for (size_t j = 0; j < output.shape()[1]; j++)
                {
                    output(i, j) += b(j);
                }
            }
            return output;
        }

        // Performs the backward pass of the dense layer
        // grad: gradient from the next layer (batch_size, out_features)
        // Returns: gradient with respect to the input (batch_size, in_features)
        utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2> &grad) override
        {
            // Calculate gradient with respect to weights (dW)
            // dW = last_x.transpose() * grad
            utec::algebra::Tensor<T, 2> last_xT = last_x.transpose_2d();
            dW = matmul(last_xT, grad);

            // Calculate gradient with respect to biases (db)
            // db = sum of grad along batch dimension
            for (size_t j = 0; j < grad.shape()[1]; j++)
            {
                db(j) = 0; // Reset db for summation
                for (size_t i = 0; i < grad.shape()[0]; i++)
                {
                    db(j) += grad(i, j);
                }
            }

            // Calculate gradient with respect to input (d_input)
            // d_input = grad * W.transpose()
            utec::algebra::Tensor<T, 2> WT = W.transpose_2d();
            return matmul(grad, WT);
        }

        // Updates the weights and biases using the learning rate
        // lr: learning rate
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

        size_t contar_parametros() const override
        {
            return W.shape()[0] * W.shape()[1] + b.shape()[0];
        }

        std::vector<T> obtener_parametros() const override
        {
            std::vector<T> params;
            params.reserve(contar_parametros());

            // Añadir pesos
            for (size_t i = 0; i < W.shape()[0]; ++i)
            {
                for (size_t j = 0; j < W.shape()[1]; ++j)
                {
                    params.push_back(W(i, j));
                }
            }

            // Añadir biases
            for (size_t i = 0; i < b.shape()[0]; ++i)
            {
                params.push_back(b(i));
            }
            return params;
        }

        void establecer_parametros(const std::vector<T> &params) override
        {
            size_t idx = 0;

            // Actualizar pesos
            for (size_t i = 0; i < W.shape()[0]; ++i)
            {
                for (size_t j = 0; j < W.shape()[1]; ++j)
                {
                    W(i, j) = params[idx++];
                }
            }

            // Actualizar biases
            for (size_t i = 0; i < b.shape()[0]; ++i)
            {
                b(i) = params[idx++];
            }
        }

    private:
        // Helper function for matrix multiplication
        // a: first matrix (rows_a, cols_a)
        // b: second matrix (rows_b, cols_b)
        // Returns: result matrix (rows_a, cols_b)
        utec::algebra::Tensor<T, 2> matmul(const utec::algebra::Tensor<T, 2> &a,
                                           const utec::algebra::Tensor<T, 2> &b) const
        {
            // Check for compatible dimensions for multiplication
            if (a.shape()[1] != b.shape()[0])
            {
                throw std::invalid_argument("Matrix dimensions must agree for multiplication");
            }

            utec::algebra::Tensor<T, 2> result(a.shape()[0], b.shape()[1]);
            for (size_t i = 0; i < a.shape()[0]; i++)
            {
                for (size_t k = 0; k < a.shape()[1]; k++)
                {
                    // Optimization: check if a(i,k) is zero to skip inner loop
                    if (a(i, k) == static_cast<T>(0))
                        continue;

                    for (size_t j = 0; j < b.shape()[1]; j++)
                    {
                        result(i, j) += a(i, k) * b(k, j);
                    }
                }
            }
            return result;
        }
    };

} // namespace utec::neural_network

#endif // UTEC_NN_DENSE_H
