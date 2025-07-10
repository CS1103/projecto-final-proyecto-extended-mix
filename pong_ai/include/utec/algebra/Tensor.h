#ifndef UTEC_ALGEBRA_TENSOR_H
#define UTEC_ALGEBRA_TENSOR_H

#include <array>
#include <vector>
#include <stdexcept>
#include <cstddef>
#include <utility>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace utec::algebra
{

    template <typename T, size_t Rank>
    class Tensor
    {
    private:
        std::array<size_t, Rank> shape_;
        std::array<size_t, Rank> strides_;
        std::vector<T> data_;

        void compute_strides() noexcept
        {
            if constexpr (Rank > 0)
            {
                strides_[Rank - 1] = 1;
                if constexpr (Rank > 1)
                {
                    for (int i = static_cast<int>(Rank) - 2; i >= 0; --i)
                    {
                        strides_[i] = strides_[i + 1] * shape_[i + 1];
                    }
                }
            }
        }

        size_t flat_index(const std::array<size_t, Rank> &indices) const
        {
            size_t index = 0;
            for (size_t i = 0; i < Rank; ++i)
            {
                if (indices[i] >= shape_[i])
                {
                    throw std::out_of_range("Index " + std::to_string(indices[i]) +
                                            " out of range for dimension " +
                                            std::to_string(i) + " (size " +
                                            std::to_string(shape_[i]) + ")");
                }
                index += indices[i] * strides_[i];
            }
            return index;
        }

        template <typename... Idxs>
        std::array<size_t, Rank> make_index_array(Idxs... idxs) const
        {
            static_assert(sizeof...(Idxs) == Rank, "Incorrect number of indices");
            std::array<size_t, Rank> indices = {static_cast<size_t>(idxs)...};
            return indices;
        }

    public:
        // Constructors
        // Default constructor - creates empty tensor
        Tensor()
        {
            shape_.fill(0);
            compute_strides();
        }

        explicit Tensor(const std::array<size_t, Rank> &shape) : shape_(shape)
        {
            size_t total_size = 1;
            for (size_t dim : shape)
            {
                total_size *= dim;
            }
            data_.resize(total_size);
            compute_strides();
        }

        template <typename... Dims>
        explicit Tensor(Dims... dims)
        {
            static_assert(sizeof...(Dims) == Rank, "Incorrect number of dimensions");
            shape_ = {static_cast<size_t>(dims)...};
            size_t total_size = 1;
            for (size_t dim : shape_)
            {
                total_size *= dim;
            }
            data_.resize(total_size);
            compute_strides();
        }

        // Access operators
        template <typename... Idxs>
        T &operator()(Idxs... idxs)
        {
            auto indices = make_index_array(idxs...);
            return data_[flat_index(indices)];
        }

        template <typename... Idxs>
        const T &operator()(Idxs... idxs) const
        {
            auto indices = make_index_array(idxs...);
            return data_[flat_index(indices)];
        }

        // Shape information
        const std::array<size_t, Rank> &shape() const noexcept
        {
            return shape_;
        }

        void reshape(const std::array<size_t, Rank> &new_shape)
        {
            size_t new_size = 1;
            for (size_t dim : new_shape)
            {
                new_size *= dim;
            }
            if (new_size != data_.size())
            {
                throw std::invalid_argument("Reshape changes total element count");
            }
            shape_ = new_shape;
            compute_strides();
        }

        // Variadic reshape
        template <typename... Dims>
        void reshape(Dims... dims)
        {
            static_assert(sizeof...(Dims) == Rank, "Incorrect number of dimensions");
            reshape(std::array<size_t, Rank>{static_cast<size_t>(dims)...});
        }

        // Bulk modification
        void fill(const T &value) noexcept
        {
            std::fill(data_.begin(), data_.end(), value);
        }

        // Arithmetic operations
        Tensor operator+(const Tensor &other) const
        {
            return binary_operation(other, [](T a, T b)
                                    { return a + b; });
        }

        Tensor operator-(const Tensor &other) const
        {
            return binary_operation(other, [](T a, T b)
                                    { return a - b; });
        }

        Tensor operator*(const Tensor &other) const
        {
            return binary_operation(other, [](T a, T b)
                                    { return a * b; });
        }

        Tensor operator*(const T &scalar) const
        {
            Tensor result(shape_);
            for (size_t i = 0; i < data_.size(); ++i)
            {
                result.data_[i] = data_[i] * scalar;
            }
            return result;
        }

        // Transpose (only for rank 2)
        Tensor transpose_2d() const
        {
            static_assert(Rank == 2, "transpose_2d requires Rank == 2");
            Tensor result(shape_[1], shape_[0]);
            for (size_t i = 0; i < shape_[0]; ++i)
            {
                for (size_t j = 0; j < shape_[1]; ++j)
                {
                    result(j, i) = (*this)(i, j);
                }
            }
            return result;
        }

    private:
        template <typename Op>
        Tensor binary_operation(const Tensor &other, Op op) const
        {
            // Check if shapes are broadcast-compatible
            std::array<size_t, Rank> result_shape;
            for (size_t i = 0; i < Rank; ++i)
            {
                if (shape_[i] == other.shape_[i])
                {
                    result_shape[i] = shape_[i];
                }
                else if (shape_[i] == 1)
                {
                    result_shape[i] = other.shape_[i];
                }
                else if (other.shape_[i] == 1)
                {
                    result_shape[i] = shape_[i];
                }
                else
                {
                    throw std::invalid_argument("Incompatible shapes for broadcasting");
                }
            }

            Tensor result(result_shape);
            std::array<size_t, Rank> indices;
            std::fill(indices.begin(), indices.end(), 0);

            // Iterate through all elements in result tensor
            for (size_t i = 0; i < result.data_.size(); ++i)
            {
                // Calculate element from this tensor
                std::array<size_t, Rank> this_indices;
                for (size_t j = 0; j < Rank; ++j)
                {
                    this_indices[j] = (shape_[j] == 1) ? 0 : indices[j];
                }
                T val1 = data_[flat_index(this_indices)];

                // Calculate element from other tensor
                std::array<size_t, Rank> other_indices;
                for (size_t j = 0; j < Rank; ++j)
                {
                    other_indices[j] = (other.shape_[j] == 1) ? 0 : indices[j];
                }
                T val2 = other.data_[other.flat_index(other_indices)];

                result.data_[i] = op(val1, val2);

                // Update indices for next element
                for (int j = Rank - 1; j >= 0; --j)
                {
                    if (++indices[j] < result_shape[j])
                        break;
                    indices[j] = 0;
                }
            }

            return result;
        }
    };

} // namespace utec::algebra

#endif // UTEC_ALGEBRA_TENSOR_H