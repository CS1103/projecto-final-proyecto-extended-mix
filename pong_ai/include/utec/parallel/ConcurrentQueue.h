#ifndef UTEC_PARALLEL_CONCURRENTQUEUE_H
#define UTEC_PARALLEL_CONCURRENTQUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <stdexcept> // For std::runtime_error

namespace utec::parallel
{

    template <typename T>
    class ConcurrentQueue
    {
    public:
        void push(const T &item)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (stop_)
            {
                throw std::runtime_error("Cannot push to a stopped queue");
            }
            queue_.push(item);
            lock.unlock();
            cond_.notify_one();
        }

        bool pop(T &item)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_.wait(lock, [this]()
                       { return !queue_.empty() || stop_; });

            if (stop_)
                return false;
            if (queue_.empty())
                return false;

            item = std::move(queue_.front());
            queue_.pop();
            return true;
        }

        void shutdown()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
            cond_.notify_all();
        }

    private:
        std::queue<T> queue_;
        std::mutex mutex_;
        std::condition_variable cond_;
        bool stop_ = false;
    };

} // namespace utec::parallel

#endif