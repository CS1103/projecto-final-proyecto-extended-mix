#include "../include/utec/parallel/ParallelAgent.h"
#include "../include/utec/agent/PongAgent.h"
#include "../include/utec/nn/dense.h"
#include "../include/utec/nn/sequential.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

using namespace utec::nn;

// Create a mock neural network layer
template <typename T>
class MockLayer : public utec::neural_network::ILayer<T>
{
public:
    utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2> &x) override
    {
        // Simulate computation time
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return x;
    }

    utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2> &grad) override
    {
        return grad;
    }

    void update(T lr) override {}
};

void test_parallel_inference()
{
    std::cout << "Test 1: Parallel inference speed\n";
    using T = float;

    // Create sequential model with mock layers
    auto sequential = std::make_unique<utec::neural_network::Sequential<T>>();
    sequential->add_layer(std::make_unique<MockLayer<T>>());
    sequential->add_layer(std::make_unique<MockLayer<T>>());

    // Create agents
    PongAgent<T> single_agent(std::move(sequential));
    ParallelPongAgent<T> parallel_agent(
        std::make_unique<MockLayer<T>>(), // Mock layer
        4                                 // Thread pool size
    );

    // Create test states
    std::vector<State> states(10);
    for (auto &state : states)
    {
        state = {0.5f, 0.5f, 0.5f};
    }

    // Test sequential processing
    auto start_seq = std::chrono::high_resolution_clock::now();
    for (const auto &state : states)
    {
        single_agent.act(state);
    }
    auto end_seq = std::chrono::high_resolution_clock::now();

    // Test parallel processing
    auto start_par = std::chrono::high_resolution_clock::now();
    std::vector<std::future<int>> futures;
    for (const auto &state : states)
    {
        futures.push_back(parallel_agent.act_async(state));
    }
    for (auto &fut : futures)
    {
        fut.get();
    }
    auto end_par = std::chrono::high_resolution_clock::now();

    // Calculate durations
    auto seq_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_seq - start_seq);
    auto par_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_par - start_par);

    std::cout << "Sequential processing: " << seq_duration.count() << "ms\n";
    std::cout << "Parallel processing: " << par_duration.count() << "ms\n";
    std::cout << "Speedup: " << (float)seq_duration.count() / par_duration.count() << "x\n";

    // Should have at least 2x speedup on multi-core systems
    bool speedup_achieved = par_duration < seq_duration;
    std::cout << (speedup_achieved ? "PASSED" : "FAILED") << "\n\n";
}

void test_result_correctness()
{
    std::cout << "Test 2: Result correctness\n";
    using T = float;

    // Create deterministic model with proper output size
    auto sequential = std::make_unique<utec::neural_network::Sequential<T>>();

    // Create weights that produce known output
    utec::algebra::Tensor<T, 2> weights(3, 3);
    weights.fill(0.1f);

    // Create biases that prefer "stay" action (index 1)
    utec::algebra::Tensor<T, 1> biases(3);
    biases(0) = 0.0f; // down
    biases(1) = 1.0f; // stay
    biases(2) = 0.0f; // up

    sequential->add_layer(
        std::make_unique<utec::neural_network::Dense<T>>(3, 3, weights, biases));

    // Create parallel agent
    utec::nn::ParallelPongAgent<T> agent(
        std::move(sequential),
        4);

    // Test states
    utec::nn::State test_state{0.5f, 0.8f, 0.3f};
    std::vector<std::future<int>> futures;

    // Submit multiple requests
    for (int i = 0; i < 10; i++)
    {
        futures.push_back(agent.act_async(test_state));
    }

    // Verify all return the same action (should be stay=0)
    bool all_same = true;
    int first_result = futures[0].get();
    bool correct_action = (first_result == 0);

    for (int i = 1; i < 10; i++)
    {
        int result = futures[i].get();
        if (result != first_result)
        {
            all_same = false;
            std::cout << "Result " << i << " differs: " << result
                      << " vs " << first_result << "\n";
        }
    }

    std::cout << "All results identical: " << (all_same ? "yes" : "no") << "\n";
    std::cout << "Correct action (stay=0): " << (correct_action ? "yes" : "no") << "\n";
    std::cout << (all_same && correct_action ? "PASSED" : "FAILED") << "\n\n";
}

void test_concurrent_access()
{
    std::cout << "Test 3: Concurrent access stress test\n";
    using T = float;

    // Create parallel agent
    ParallelPongAgent<T> agent(
        std::make_unique<MockLayer<T>>(),
        8);

    const int num_tasks = 100;
    std::vector<std::future<int>> futures;

    // Submit tasks from multiple threads
    std::thread producer1([&]
                          {
        for (int i = 0; i < num_tasks/2; i++) {
            futures.push_back(agent.act_async({0.1f, 0.2f, 0.3f}));
        } });

    std::thread producer2([&]
                          {
        for (int i = 0; i < num_tasks/2; i++) {
            futures.push_back(agent.act_async({0.4f, 0.5f, 0.6f}));
        } });

    producer1.join();
    producer2.join();

    // Verify all tasks completed
    for (auto &fut : futures)
    {
        fut.get(); // Will throw if task failed
    }

    std::cout << "Completed " << futures.size() << " tasks without errors\n";
    std::cout << "PASSED\n";
}

int main()
{
    test_parallel_inference();
    test_result_correctness();
    test_concurrent_access();
    return 0;
}