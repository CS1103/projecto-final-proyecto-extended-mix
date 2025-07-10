#ifndef UTEC_AGENT_PONGAGENT_H
#define UTEC_AGENT_PONGAGENT_H

#include "../nn/neural_network.h"
#include "EnvGym.h"
#include "State.h"
#include <memory>
#include <stdexcept>

namespace utec::nn
{

    template <typename T>
    class PongAgent
    {
    private:
        std::unique_ptr<utec::neural_network::ILayer<T>> model;

    public:
        PongAgent(std::unique_ptr<utec::neural_network::ILayer<T>> m)
            : model(std::move(m)) {}

        int act(const State &s)
        {
            // Convert state to tensor (batch size 1, 3 features)
            utec::algebra::Tensor<T, 2> input(1, 3);
            input(0, 0) = static_cast<T>(s.ball_x);
            input(0, 1) = static_cast<T>(s.ball_y);
            input(0, 2) = static_cast<T>(s.paddle_y);

            // Forward pass through the model
            utec::algebra::Tensor<T, 2> output = model->forward(input);

            // Handle different output dimensions
            if (output.shape()[1] < 3)
            {
                throw std::runtime_error("Model output must have at least 3 columns");
            }

            // Get action with highest probability in the first 3 columns
            int action_index = 0;
            T max_val = output(0, 0);

            for (int i = 1; i < 3; i++)
            {
                if (output(0, i) > max_val)
                {
                    max_val = output(0, i);
                    action_index = i;
                }
            }

            // Map to action: 0 = down, 1 = stay, 2 = up
            // But the test expects: +1 = up, -1 = down
            if (action_index == 0)
                return 1; // down
            if (action_index == 2)
                return -1; // up
            return 0;      // stay
        }

        // Nuevos métodos
        std::vector<T> obtener_parametros()
        {
            return model->obtener_parametros();
        }

        void establecer_parametros(const std::vector<T> &params)
        {
            model->establecer_parametros(params);
        }
    };

} // namespace utec::nn

#endif // UTEC_AGENT_PONGAGENT_H