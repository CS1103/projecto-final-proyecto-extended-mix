#ifndef UTEC_PARALLEL_PARALLELAGENT_H
#define UTEC_PARALLEL_PARALLELAGENT_H

#include "ThreadPool.h"
#include "../agent/PongAgent.h"

using namespace utec::neural_network;

namespace utec::nn
{

    template <typename T>
    class ParallelPongAgent
    {
    public:
        ParallelPongAgent(std::unique_ptr<ILayer<T>> model, size_t pool_size = 4)
            : agent_(std::move(model)), pool_(pool_size)
        {
        }

        std::future<int> act_async(const State &state)
        {
            return pool_.enqueue([this, state]
                                 { return agent_.act(state); });
        }

        int act(const State &state)
        {
            return agent_.act(state);
        }

    private:
        PongAgent<T> agent_;
        utec::parallel::ThreadPool pool_;
    };

} // namespace utec::nn

#endif