# Deep RL Projects

Implementation of deep reinforcement learning models

### Models
- Soft Actor Critic
- GAIL (In Progress)

#### SAC
Soft actor critic is an off-policy model that attempts to maximize reward as well as entropy of its actions. With its objective being</br>
<img src="https://render.githubusercontent.com/render/math?math=J(\theta) = E\[\sum_t r(s_t, a_t) - \alpha * \log(\pi(a_t|s_t))\]"></br>
This pushes the policy to balance between exploration and exploitation of its environment with minimum number of hyperparameters to tune.</br>
The policy uses a gaussian distribution for continuous action prediction and the value network uses a twin q-net to prevent explosive growth in reward.
