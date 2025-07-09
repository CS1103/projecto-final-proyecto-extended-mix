#ifndef UTEC_NN_LAYER_H
#define UTEC_NN_LAYER_H

#include "../algebra/Tensor.h"

using namespace utec::algebra;

namespace utec::neural_network
{

    template <typename T>
    class ILayer
    {
    public:
        virtual ~ILayer() = default;
        virtual Tensor<T, 2> forward(const Tensor<T, 2> &x) = 0;
        virtual Tensor<T, 2> backward(const Tensor<T, 2> &grad) = 0;
        virtual void update(T lr) = 0;
    };

} // namespace utec::neural_network

#endif // UTEC_NN_LAYER_H