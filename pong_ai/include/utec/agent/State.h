#ifndef UTEC_AGENT_STATE_H
#define UTEC_AGENT_STATE_H

namespace utec::nn
{
    struct State
    {
        float ball_x, ball_y;
        float paddle_y;
    };
}

#endif