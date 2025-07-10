#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <sstream>
#include "../include/utec/nn/neural_network.h"
#include "../include/utec/nn/dense.h"
#include "../include/utec/nn/activation.h"
#include "../include/utec/nn/loss.h"
#include "../include/utec/nn/sequential.h"
#include "../include/utec/agent/PongAgent.h"
#include "../include/utec/agent/EnvGym.h"

using namespace utec::neural_network;
using namespace utec::nn;
using namespace utec::algebra;

// Function to read CSV files into vectors
std::vector<std::vector<float>> read_input_csv(const std::string &filename)
{
    std::vector<std::vector<float>> data;
    std::ifstream file(filename);

    if (!file)
    {
        std::cerr << "Error: Could not open input file: " << filename << "\n";
        return data;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ','))
        {
            try
            {
                row.push_back(std::stof(value));
            }
            catch (...)
            {
                std::cerr << "Warning: Invalid float value in input CSV: '" << value << "'\n";
                row.clear();
                break;
            }
        }

        // Verify exactly 3 values per row (ball_x, ball_y, paddle_y)
        if (row.size() == 3)
        {
            data.push_back(row);
        }
        else if (!row.empty())
        {
            std::cerr << "Warning: Expected 3 values per row, got " << row.size() << "\n";
        }
    }

    if (data.empty())
    {
        std::cerr << "Error: No valid data found in input file\n";
    }
    else
    {
        std::cout << "Successfully loaded " << data.size() << " input samples\n";
        std::cout << "First sample: "
                  << data[0][0] << ", "
                  << data[0][1] << ", "
                  << data[0][2] << "\n";
    }

    return data;
}

// Function to convert action value to one-hot encoding
std::vector<float> action_to_onehot(int action)
{
    std::vector<float> onehot(3, 0.0f);
    if (action == -1)
        onehot[2] = 1.0f; // Up
    else if (action == 0)
        onehot[1] = 1.0f; // Stay
    else if (action == 1)
        onehot[0] = 1.0f; // Down
    return onehot;
}

// L2 Regularization implementation
void apply_l2_regularization(NeuralNetwork<float> &net, float lambda)
{
    auto params = net.obtener_parametros();
    for (auto &param : params)
    {
        param -= lambda * param;
    }
    net.establecer_parametros(params);
}

// Magnitude-based pruning
void prune_network(NeuralNetwork<float> &net, float prune_ratio)
{
    auto params = net.obtener_parametros();

    // Calculate threshold based on magnitude
    std::vector<float> abs_params;
    for (auto p : params)
        abs_params.push_back(std::abs(p));
    std::sort(abs_params.begin(), abs_params.end());
    float threshold = abs_params[static_cast<size_t>(prune_ratio * abs_params.size())];

    // Prune parameters below threshold
    for (auto &p : params)
    {
        if (std::abs(p) < threshold)
            p = 0.0f;
    }

    net.establecer_parametros(params);
}

// Function to compute accuracy
float compute_accuracy(const Tensor<float, 2> &pred, const Tensor<float, 2> &Y)
{
    size_t num_samples = pred.shape()[0];
    size_t correct = 0;

    for (size_t i = 0; i < num_samples; ++i)
    {
        int pred_action = 0;
        int true_action = 0;
        float max_val = pred(i, 0);

        // Find predicted action
        for (int j = 1; j < 3; j++)
        {
            if (pred(i, j) > max_val)
            {
                max_val = pred(i, j);
                pred_action = j;
            }
        }

        // Find true action
        if (Y(i, 1) == 1.0f)
            true_action = 1;
        else if (Y(i, 2) == 1.0f)
            true_action = 2;

        if (pred_action == true_action)
            correct++;
    }

    return static_cast<float>(correct) / num_samples * 100.0f;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " input.csv [output.csv]\n";
        return 1;
    }

    // Load only input data
    const std::string input_file = argv[1];
    auto input_data = read_input_csv(input_file);
    if (input_data.empty())
    {
        return 1;
    }

    // Prepare input tensor
    size_t num_samples = input_data.size();
    Tensor<float, 2> X(num_samples, 3); // Input features

    for (size_t i = 0; i < num_samples; ++i)
    {
        // Input features
        for (size_t j = 0; j < 3; ++j)
        {
            X(i, j) = input_data[i][j];
        }
    }

    // Create network architecture
    Sequential<float> model;
    model.add_layer(std::make_unique<Dense<float>>(3, 64)); // Input: 3 features
    model.add_layer(std::make_unique<ReLU<float>>());
    model.add_layer(std::make_unique<Dense<float>>(64, 32));
    model.add_layer(std::make_unique<ReLU<float>>());
    model.add_layer(std::make_unique<Dense<float>>(32, 3)); // Output: 3 actions

    // Create neural network
    NeuralNetwork<float> net;
    net.add_layer(std::make_unique<Sequential<float>>(std::move(model)));

    // Training parameters
    const size_t epochs = 1000;
    const float learning_rate = 0.01f;
    const float l2_lambda = 0.001f; // L2 regularization strength
    const float prune_ratio = 0.1f; // Prune 10% of smallest weights

    // Open results file for Colab monitoring
    const std::string output_file = (argc > 2) ? argv[2] : "output.csv";
    std::ofstream results_file(output_file);
    results_file << "epoch,reward,precision\n";

    // Training loop
    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        // Forward pass
        Tensor<float, 2> pred = net.forward(X);

        // Generate synthetic targets and calculate accuracy
        Tensor<float, 2> Y(num_samples, 3);
        size_t correct = 0;
        for (size_t i = 0; i < num_samples; ++i)
        {
            float ball_y = X(i, 1);
            float paddle_y = X(i, 2);
            float diff = ball_y - paddle_y;

            // Determine correct action
            int true_action;
            if (diff > 0.1f)
            {
                Y(i, 0) = 1.0f; // down
                true_action = 0;
            }
            else if (diff < -0.1f)
            {
                Y(i, 2) = 1.0f; // up
                true_action = 2;
            }
            else
            {
                Y(i, 1) = 1.0f; // stay
                true_action = 1;
            }

            // Calculate accuracy
            int pred_action = 0;
            float max_val = pred(i, 0);
            for (int j = 1; j < 3; j++)
            {
                if (pred(i, j) > max_val)
                {
                    max_val = pred(i, j);
                    pred_action = j;
                }
            }
            if (pred_action == true_action)
                correct++;
        }
        float accuracy = static_cast<float>(correct) / num_samples * 100.0f;

        // Calculate loss
        MSELoss<float> criterion;
        float loss = criterion.forward(pred, Y);

        // Backward pass
        Tensor<float, 2> grad = criterion.backward();
        net.backward(grad);
        net.optimizer(learning_rate);

        // Apply L2 regularization
        apply_l2_regularization(net, l2_lambda);

        // Colab monitoring output
        if (epoch % 10 == 0)
        {
            std::cout << "Epoch " << epoch << " | Best Reward: " << (100 - loss)
                      << " | Precision: " << accuracy << "%\n";

            // Save to CSV for final analysis
            results_file << epoch << "," << (100 - loss) << "," << accuracy << "\n";
        }
    }

    // Apply pruning for parameter reduction
    prune_network(net, prune_ratio);
    std::cout << "Applied pruning: Removed " << prune_ratio * 100
              << "% of smallest weights" << std::endl;

    // Save trained parameters
    auto params = net.obtener_parametros();
    std::ofstream param_file("trained_params.txt");
    for (const auto &p : params)
    {
        param_file << p << "\n";
    }
    param_file.close();

    std::cout << "Training complete! Parameters saved to trained_params.txt" << std::endl;
    results_file.close();

    return 0;
}