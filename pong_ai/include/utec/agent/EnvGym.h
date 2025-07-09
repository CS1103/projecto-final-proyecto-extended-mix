#ifndef UTEC_AGENT_ENVGYM_H
#define UTEC_AGENT_ENVGYM_H

#include <random>
#include "State.h"

namespace utec::nn
{

    class EnvGym
    {
    private:
        // Game state
        float ball_x, ball_y;
        float ball_vx, ball_vy;
        float paddle_y;
        bool done_flag;

        // Constants
        const float paddle_height = 0.2f;
        const float paddle_width = 0.02f;
        const float ball_radius = 0.02f;
        const float dt = 0.016f; // ~60 FPS

        // Random number generation
        std::mt19937 rng;
        std::uniform_real_distribution<float> vel_dist;

    public:
        EnvGym() : rng(std::random_device{}()),
                   vel_dist(-0.05f, 0.05f)
        {
            // nop
        }

        State reset()
        {
            // Initialize ball at center
            ball_x = 0.5f;
            ball_y = 0.5f;

            // Random initial velocity (mostly rightward)
            ball_vx = 0.03f + vel_dist(rng);
            ball_vy = vel_dist(rng);

            paddle_y = 0.5f;
            done_flag = false;
            return get_state();
        }

        State step(int action, float &reward, bool &done)
        {
            if (done_flag)
            {
                reward = 0;
                done = true;
                return get_state();
            }

            // Update paddle position based on action
            paddle_y += action * 0.04f;
            paddle_y = std::max(0.1f, std::min(0.9f, paddle_y));

            // Update ball position
            ball_x += ball_vx;
            ball_y += ball_vy;

            // Handle collisions
            reward = 0;
            done = false;

            // Top/bottom walls
            if (ball_y <= ball_radius || ball_y >= 1.0f - ball_radius)
            {
                ball_vy = -ball_vy;
                ball_y = std::clamp(ball_y, ball_radius, 1.0f - ball_radius);
            }

            // Right wall (paddle)
            if (ball_x >= 1.0f - paddle_width - ball_radius)
            {
                float paddle_top = paddle_y - paddle_height / 2;
                float paddle_bottom = paddle_y + paddle_height / 2;

                if (ball_y >= paddle_top && ball_y <= paddle_bottom)
                {
                    // Successful hit
                    ball_vx = -ball_vx * 1.05f;            // Speed up slightly
                    ball_vy += (ball_y - paddle_y) * 0.5f; // Add spin
                    ball_x = 1.0f - paddle_width - ball_radius - 0.001f;
                    reward = 1.0f;
                }
                else if (ball_x >= 1.0f)
                {
                    // Missed the ball
                    reward = -1.0f;
                    done_flag = true;
                    done = true;
                }
            }

            // Left wall (opponent)
            if (ball_x <= ball_radius)
            {
                ball_vx = -ball_vx;
                ball_x = ball_radius + 0.001f;
            }

            return get_state();
        }

    private:
        State get_state() const
        {
            return {ball_x, ball_y, paddle_y};
        }
    };

} // namespace utec::nn
#endif // UTEC_AGENT_ENVGYM_H