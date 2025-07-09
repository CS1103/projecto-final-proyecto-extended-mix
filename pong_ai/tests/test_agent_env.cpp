#include "../include/utec/agent/PongAgent.h"
#include "../include/utec/agent/EnvGym.h"
#include "../include/utec/nn/dense.h"
#include "../include/utec/nn/sequential.h"
#include <iostream>

using namespace utec::nn;

// Helper to create a deterministic model that prefers down action
template <typename T>
std::unique_ptr<utec::neural_network::Sequential<T>> create_down_model()
{
    auto sequential = std::make_unique<utec::neural_network::Sequential<T>>();

    // Create weights that always prefer down action (index 0)
    utec::algebra::Tensor<T, 2> weights(3, 3);
    weights.fill(0.0f);

    // Create biases that prefer down action
    utec::algebra::Tensor<T, 1> biases(3);
    biases(0) = 1.0f;  // down
    biases(1) = 0.0f;  // stay
    biases(2) = -1.0f; // up

    sequential->add_layer(
        std::make_unique<utec::neural_network::Dense<T>>(3, 3, weights, biases));
    return sequential;
}

void test_basic_instantiation()
{
    std::cout << "1. Instanciación básica\n";
    using T = float;
    auto sequential = create_down_model<T>();
    auto agent = PongAgent<T>(std::move(sequential));

    State s{0.5f, 0.8f, 0.3f};
    int a = agent.act(s);
    std::cout << "Action: " << a << " (expected 1)\n";
    std::cout << (a == 1 ? "PASSED" : "FAILED") << "\n\n";
}

void test_single_step()
{
    std::cout << "2. Simulación de un paso\n";
    EnvGym env;
    float reward;
    bool done;
    auto s0 = env.reset();

    using T = float;
    auto sequential = create_down_model<T>();
    auto agent = PongAgent<T>(std::move(sequential));

    int a0 = agent.act(s0);
    auto s1 = env.step(a0, reward, done);
    std::cout << "Estado: " << s1.ball_x << "," << s1.ball_y
              << " | Recompensa: " << reward
              << " | Done: " << done << "\n";
    std::cout << "TEST EXECUTED (manual verification required)\n\n";
}

void test_integration()
{
    std::cout << "3. Integración agent + entorno\n";
    EnvGym env;
    using T = float;
    auto sequential = create_down_model<T>();
    auto agent = PongAgent<T>(std::move(sequential));

    auto s0 = env.reset();
    float reward;
    bool done;

    for (int t = 0; t < 5; ++t)
    {
        int a = agent.act(s0);
        s0 = env.step(a, reward, done);
        std::cout << "Step " << t
                  << " action=" << a
                  << " reward=" << reward
                  << "\n";
        if (done)
            break;
    }
    std::cout << "TEST EXECUTED (manual verification required)\n\n";
}

void test_boundaries()
{
    std::cout << "4. Prueba de límites\n";
    using T = float;

    // Create a model that prefers staying
    auto sequential = std::make_unique<utec::neural_network::Sequential<T>>();
    utec::algebra::Tensor<T, 2> weights(3, 3);
    weights.fill(0.0f);
    utec::algebra::Tensor<T, 1> biases(3);
    biases(0) = -1.0f; // down
    biases(1) = 1.0f;  // stay
    biases(2) = -1.0f; // up
    sequential->add_layer(std::make_unique<utec::neural_network::Dense<T>>(3, 3, weights, biases));

    auto agent = PongAgent<T>(std::move(sequential));

    State eq{0.2f, 0.5f, 0.5f};
    int action = agent.act(eq);
    std::cout << "Action at boundary: " << action << " (expected 0)\n";
    bool test = (action == 0);
    std::cout << (test ? "PASSED" : "FAILED") << "\n";
}

int main()
{
    test_basic_instantiation();
    test_single_step();
    test_integration();
    test_boundaries();
    return 0;
}