#ifndef UTEC_PARALLEL_THREADPOOL_H
#define UTEC_PARALLEL_THREADPOOL_H

#include "ConcurrentQueue.h"
#include <vector>
#include <thread>
#include <functional>
#include <future>

namespace utec::parallel
{

    class ThreadPool
    {
    public:
        explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency())
            : queue_(), workers_()
        {
            for (size_t i = 0; i < num_threads; ++i)
            {
                workers_.emplace_back([this]
                                      {
                while (true) {
                    std::function<void()> task;
                    if (!queue_.pop(task)) break;
                    task();
                } });
            }
        }

        ~ThreadPool()
        {
            queue_.shutdown();
            for (auto &worker : workers_)
            {
                if (worker.joinable())
                    worker.join();
            }
        }

        template <typename F, typename... Args>
        auto enqueue(F &&f, Args &&...args) -> std::future<decltype(f(args...))>
        {
            using return_type = decltype(f(args...));

            auto task = std::make_shared<std::packaged_task<return_type()>>(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...));

            std::future<return_type> res = task->get_future();
            queue_.push([task]()
                        { (*task)(); });
            return res;
        }

    private:
        ConcurrentQueue<std::function<void()>> queue_;
        std::vector<std::thread> workers_;
    };

} // namespace utec::parallel

#endif