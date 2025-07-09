#include "../include/utec/nn/layer.h"
#include "../include/utec/nn/dense.h"
#include "../include/utec/nn/activation.h"
#include "../include/utec/nn/loss.h"
#include "../include/utec/nn/neural_network.h"
#include <iostream>

using namespace utec::neural_network;

void test_relu()
{
    std::cout << "Prueba ReLU forward/backward\n";
    Tensor<float, 2> M(2, 2);
    M(0, 0) = -1;
    M(0, 1) = 2;
    M(1, 0) = 0;
    M(1, 1) = -3;

    auto relu = ReLU<float>();
    auto R = relu.forward(M);
    bool test1 = R(0, 1) == 2;

    Tensor<float, 2> GR({2, 2});
    GR.fill(1.0f);
    auto dM = relu.backward(GR);
    bool test2 = dM(0, 0) == 0 && dM(1, 1) == 0;

    std::cout << (test1 && test2 ? "PASSED" : "FAILED") << "\n\n";
}

void test_mseloss()
{
    std::cout << "Prueba MSELoss forward/backward\n";
    utec::algebra::Tensor<double, 2> P(1, 2);
    P(0, 0) = 1;
    P(0, 1) = 2;

    utec::algebra::Tensor<double, 2> Tgt(1, 2);
    Tgt(0, 0) = 0;
    Tgt(0, 1) = 4;
    auto loss = MSELoss<double>();
    double L = loss.forward(P, Tgt);
    bool test1 = std::abs(L - 2.5) < 1e-6;
    Tensor<double, 2> dP = loss.backward();
    bool test2 = std::abs(dP(0, 1) - (-2.0)) < 1e-6;
    std::cout << (test1 && test2 ? "PASSED" : "FAILED") << "\n\n";
}

void test_xor()
{
    std::cout << "Prueba Entrenamiento XOR\n";
    using T = float;

    // XOR data
    utec::algebra::Tensor<T, 2> X(4, 2);
    X(0, 0) = 0;
    X(0, 1) = 0;
    X(1, 0) = 0;
    X(1, 1) = 1;
    X(2, 0) = 1;
    X(2, 1) = 0;
    X(3, 0) = 1;
    X(3, 1) = 1;

    utec::algebra::Tensor<T, 2> Y(4, 1);
    Y(0, 0) = 0;
    Y(1, 0) = 1;
    Y(2, 0) = 1;
    Y(3, 0) = 0;

    utec::neural_network::NeuralNetwork<T> net;
    net.add_layer(std::make_unique<utec::neural_network::Dense<T>>(2, 4));
    net.add_layer(std::make_unique<utec::neural_network::ReLU<T>>());
    net.add_layer(std::make_unique<utec::neural_network::Dense<T>>(4, 1));

    // Use better training parameters for XOR
    float final_loss = net.train(X, Y, 10000, 0.1f);
    std::cout << "Final loss: " << final_loss << "\n";

    // Test predictions
    try
    {
        auto pred = net.forward(X);
        std::cout << "Predictions vs Expected:\n";
        for (size_t i = 0; i < 4; i++)
        {
            std::cout << "[" << pred(i, 0) << " vs " << Y(i, 0) << "] ";
        }
        std::cout << "\n";

        // Check if predictions are correct
        bool passed = true;
        for (size_t i = 0; i < 4; i++)
        {
            T prediction = pred(i, 0) > 0.5 ? 1 : 0;
            if (std::abs(prediction - Y(i, 0)) > 0.1)
            {
                passed = false;
            }
        }

        std::cout << (passed ? "PASSED" : "FAILED") << "\n\n";
    }
    catch (const std::exception &e)
    {
        std::cout << "EXCEPTION during prediction: " << e.what() << "\n";
        std::cout << "FAILED\n\n";
    }
}

void test_shape_mismatch()
{
    std::cout << "Prueba Shape mismatch\n";
    using T = float; // Add this line to specify the type

    bool passed = false;
    NeuralNetwork<T> net;
    net.add_layer(std::make_unique<Dense<T>>(2, 4));
    net.add_layer(std::make_unique<ReLU<T>>());
    net.add_layer(std::make_unique<Dense<T>>(4, 1));

    try
    {
        // Create a tensor with wrong dimensions (3 features instead of 2)
        Tensor<T, 2> input(3, 3); // Should be (batch, 2)
        net.forward(input);
    }
    catch (const std::invalid_argument &e)
    {
        std::cout << "Caught exception: " << e.what() << "\n";
        passed = true;
    }
    catch (const std::exception &e)
    {
        std::cout << "Caught wrong exception: " << e.what() << "\n";
    }

    std::cout << (passed ? "PASSED" : "FAILED") << "\n\n";
}

int main()
{
    test_relu();
    test_mseloss();
    test_xor();
    test_shape_mismatch();
    return 0;
}