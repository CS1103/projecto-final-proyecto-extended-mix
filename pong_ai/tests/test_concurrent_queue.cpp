#include "../include/utec/parallel/ConcurrentQueue.h"
#include <thread>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace utec::parallel;

void test_basic_operations()
{
    std::cout << "Test 1: Basic queue operations\n";
    ConcurrentQueue<int> queue;

    // Push and pop single item
    queue.push(42);
    int value = 0;
    bool success = queue.pop(value);

    std::cout << "Pop result: " << (success ? "success" : "fail")
              << ", value: " << value << " (expected 42)\n";
    std::cout << (success && value == 42 ? "PASSED" : "FAILED") << "\n\n";
}

void test_concurrent_producer_consumer()
{
    std::cout << "Test 2: Concurrent producer-consumer\n";
    ConcurrentQueue<int> queue;
    const int num_items = 1000;
    std::vector<int> consumed;

    // Consumer thread
    std::thread consumer([&]
                         {
        for (int i = 0; i < num_items; ) {
            int value;
            if (queue.pop(value)) {
                consumed.push_back(value);
                i++;
            }
        } });

    // Producer thread
    std::thread producer([&]
                         {
        for (int i = 0; i < num_items; i++) {
            queue.push(i);
        } });

    producer.join();
    consumer.join();

    // Verify all items were processed
    bool all_present = true;
    for (int i = 0; i < num_items; i++)
    {
        if (std::find(consumed.begin(), consumed.end(), i) == consumed.end())
        {
            all_present = false;
            break;
        }
    }

    std::cout << "Items produced: " << num_items
              << ", consumed: " << consumed.size()
              << ", all present: " << (all_present ? "yes" : "no") << "\n";
    std::cout << (consumed.size() == num_items && all_present ? "PASSED" : "FAILED") << "\n\n";
}

void test_shutdown_behavior()
{
    std::cout << "Test 3: Queue shutdown behavior\n";
    ConcurrentQueue<int> queue;

    // Start consumer before shutdown
    std::thread consumer([&]
                         {
        int value;
        bool result = queue.pop(value);
        std::cout << "Pop after shutdown: " << (result ? "success" : "fail") << "\n";
        std::cout << (result ? "FAILED" : "PASSED") << " - should fail after shutdown\n"; });

    // Give consumer time to block
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Shutdown queue
    queue.shutdown();
    consumer.join();

    // Test push after shutdown
    try
    {
        queue.push(42);
        std::cout << "FAILED - Push after shutdown should throw\n";
    }
    catch (const std::runtime_error &e)
    {
        std::cout << "Caught exception: " << e.what() << "\n";
        std::cout << "PASSED - Push after shutdown throws exception\n";
    }
    catch (...)
    {
        std::cout << "FAILED - Wrong exception type\n";
    }
}

int main()
{
    test_basic_operations();
    test_concurrent_producer_consumer();
    test_shutdown_behavior();
    return 0;
}