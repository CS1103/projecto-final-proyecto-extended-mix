#include "../include/utec/parallel/ThreadPool.h"
#include <iostream>
#include <vector>
#include <atomic>
#include <future>

using namespace utec::parallel;

void test_basic_task_execution()
{
    std::cout << "Test 1: Basic task execution\n";
    ThreadPool pool(2);
    std::atomic<int> counter(0);

    // Submit tasks to increment counter
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 10; i++)
    {
        futures.push_back(pool.enqueue([&]
                                       { counter++; }));
    }

    // Wait for all tasks to complete
    for (auto &fut : futures)
    {
        fut.get();
    }

    std::cout << "Final counter: " << counter << " (expected 10)\n";
    std::cout << (counter == 10 ? "PASSED" : "FAILED") << "\n\n";
}

void test_parallel_computation()
{
    std::cout << "Test 2: Parallel computation\n";
    ThreadPool pool(4);
    const int num_tasks = 8;
    std::vector<std::future<int>> futures;

    // Submit compute-intensive tasks
    for (int i = 0; i < num_tasks; i++)
    {
        futures.push_back(pool.enqueue([i]
                                       {
            // Simulate work
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return i * i; }));
    }

    // Verify results
    bool all_correct = true;
    for (int i = 0; i < num_tasks; i++)
    {
        int result = futures[i].get();
        if (result != i * i)
        {
            all_correct = false;
            std::cout << "Task " << i << " returned " << result
                      << ", expected " << i * i << "\n";
        }
    }

    std::cout << "All results correct: " << (all_correct ? "yes" : "no") << "\n";
    std::cout << (all_correct ? "PASSED" : "FAILED") << "\n\n";
}

void test_exception_handling()
{
    std::cout << "Test 3: Exception handling\n";
    ThreadPool pool(2);

    auto future = pool.enqueue([]
                               {
        throw std::runtime_error("Test exception");
        return 42; });

    try
    {
        int result = future.get();
        std::cout << "FAILED - Exception not propagated\n";
    }
    catch (const std::runtime_error &e)
    {
        std::cout << "Caught exception: " << e.what() << "\n";
        std::cout << "PASSED - Exception propagated correctly\n";
    }
    catch (...)
    {
        std::cout << "FAILED - Wrong exception type\n";
    }
}

int main()
{
    test_basic_task_execution();
    test_parallel_computation();
    test_exception_handling();
    return 0;
}