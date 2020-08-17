# Report.md

# This is still a manuscript!!

## Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

### MADDPG paper

R. Lowe et al., 2017. *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments* (https://arxiv.org/abs/1706.02275)


### The basics of MADDPG


### Neural network architectures

**The actor**

The actor network learns the policy of the agent. In my implementation, each agent has its own actor network. The input to the actor network
is the state vector which is only observed by the agent (size = 24), and the output will be the action taken by the agent (size = 2). The action 
takes continuous real values.

The default architectures of the actor (both local and target) network consist of:

- **input layer**: a fully connected layer with input size = 24, output size = 400, activation function = ReLU
- **hidden layer**: a fully connected layer with input size = 400, output size = 300, activation function = ReLU
- **output layer**: a fully connected layer with input size = 300, output size = 2, activation function = tanh

**The critic**

The critic network learns how good/bad the actor's policy is. In the implementation, each agent has its own critic network. The critic network takes
the **states observed by both agents** and the **actions taken by both agents** as the inputs, and outputs an action-value **q(s, a)**

The default architectures of the critic (both local and target) network consist of:

- **input layer**: a fully connected layer with input size = 48, output size = 400, activation function = ReLU
- **hidden layer**: a fully connected layer with input size = 404, output size = 300, activation function = ReLU
- **output layer**: a fully connected layer with input size = 300, output size = 1

**Notes** I used the architecture used in the DDPG paper (https://arxiv.org/abs/1509.02971), the state vector (size = 48) is passed through the input layer alone.
After the ReLU activation, the output (size = 400) is concatenated with the action vector (size = 4) and becomes the input (size = 404) to the next hidden layer.

More details can be found in the file `networkModels.py`


### The noise process
The essence of policy-based methods is that, to carry out the exploration, the agent needs to try some new actions that are slightly different from the one prescribed 
by the current policy. In the case where the action space is continuous, this can simply be achieved by calculating the action by the actor network and then adding 
some noise on it. The magnitude of the noise determines how far away from the current policy that the agent can explore for any putative better policies yielding 
higher rewards.

In the **DDPG paper**, the authors modeled the noise by the **Ornstein-Uhlenbeck Process** (OU noise)

**why OU noise?**
In a nutshell, compared to the Gaussian noise, which consists of pure diffusion, the Ornstein-Uhlenbeck process has an additional **drift** component. If the direction 
of the drift is set to be always pointing toward the asymptotic mean of the noise, it can be pictured as a drag that prevents the noise from diffusing 
to +/- infinity. (note that in the case of Gaussian noise, as time increases to infinity, the noise will also diffuse to +/- infinity)

The OU noise can be modeled by 2 parameters, `ou_theta` that controls the magnitude of the drift term, and `ou_sigma` that controls the magnitude of the diffusion term. 
In the **DDPG paper**, the authors used `ou_theta` = 0.15 and `ou_sigma` = 0.20. In this project I used `ou_theta` = 0.15 and `ou_sigma` = 0.10. Moreover, I slowly 
decreased the scaling factor `ou_scale` (starting from 1.0) of the noise, by a decay rate `ou_decay` = 0.995 after each episode.

For more details of the Ornstein-Uhlenbeck Process, please refer to the textbook *Stochastic Methods, a handbook for the natural and social sciences* 
by Crispin Gardiner (https://www.springer.com/gp/book/9783540707127).


### The MADDPG algorithm

Below is a brief description of MADDPG:

- initialize the 2 agents with the actor (local and target) and critic (local and target) networks
- t_step = 0
- each agent observes the initial state **s_i** (size = 2*24 = 48) from the environment
- while the environment is not solved:
  - t_step += 1
  - each agent uses *actor_local* to determine the action from its own observation: `**a_i** = actor_local(s_i)`
  - each agent uses **a_i** to interact with the environment
  - each agent collects the reward **r_i** and enters the next state (**s'_i**)
  - combine the **s_i**, **a_i**, **r_i**, and **s'_i** of each agent into vectors **s**, **a**, **r**, and **s'** respectively
  - add the experience tuple **(s, a, r, s')** into the replay buffer
  
  - if there are enough (>= `batch_size`) replays in the buffer:
    - for each agent:
        **step 1: sample replays**
        - randomly sample a batch of size = `batch_size` replay tuples **(s, a, r, s')** from the buffer

        **step 2: train `critic_local`**
        - use *actor_target* predict the future actions **a'** = actor_target(s')
        - use *critic_target* to predict the action value of the next state **q_next** = critic_target(s', a')
        - calculate the TD target of the action value **q_target** = **r** + gamma * **q_next**
        - use *critic_local* to calculate the current action value **q_local** = critic_local(s, a)
        - define the loss function (Huber loss) to be the TD error, i.e.,\
        `critic_loss = F.smooth_l1_loss(q_local, q_target)`
        - use gradient decent to update the weights of *critic_local*

        **step 3: train `actor_local`**
        - use *actor_local* to determine the action **a_local** = actor_local(s)
        - use *critic_local* to calculate the action values and averaged over the sample batch to obtain the loss of actor:\
        `actor_loss = -critic_local(s, a_local).mean()` (averaged over the batch)
        - use gradient descent to update the weights of *actor_local*

        **step 4: update `actor_target` and `critic_target`**
        - soft-update the weights of *actor_target* and *critic_target*\
        `actor_target.weights <-- tau * actor_local.weights + (1-tau) * actor_target.weights`\
        `critic_target.weights <-- tau * critic_local.weights + (1-tau) * critic_target.weights`
  - **s** <-- **s'**


## Hyperparameters

| Hyperparameter | Value | Description | Reference |
| ----------- | ----------- | ----------- | ----------- |
| actor_hidden_sizes | [400, 300] | sizes of actor's hidden FC layers | <1> |
| critic_hidden_sizes | [400, 300] | sizes of critic's the hidden FC layers | <1> |
| gamma | 0.99 | discount rate | <1> |
| actor_lr | 1e-3 | actor's learning rate |  |
| critic_lr | 1e-4 | critic's learning rate |  |
| critic_L2_decay | 0 | critic's L2 weight decay |  |
| tau | 1e-3 | soft update | <1> |
| ou_scale | 1.0 | initial scaling of noise |  |
| ou_decay | 0.995 | scaling decay of noise |  |
| ou_mu | 0.0 | initial mean of noise | <1> |
| ou_theta | 0.15 | drift component of noise | <1> |
| ou_sigma | 0.20 | diffusion component of noise | <1> |
| buffer_size | 1e6 | size of the replay buffer | <1> |
| batch_size | 256 | size of the minibatch |  |

<1> Theses are the values used in the original DDPG paper (https://arxiv.org/pdf/1509.02971.pdf)
Please refer to the **Experiment details** in the paper for more information.


## Training result

With the parameters above, the agent solved the task after N/A episodes, i.e., the average (over agents) score from episode #N/A to #N/A reaches above +30.0 points.

**(figure)** *Average score of each episode*


## Ideas for future work

Multi-agent problems consists of not only collaboration tasks but also competition tasks. How to train the 2 agents to compete with each other?

## References in this project
1. R. Lowe et al., 2017. *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*\
https://arxiv.org/abs/1706.02275
2. T. P. Lillicrap et al., 2016. *Continuous control with deep reinforcement learning*\
https://arxiv.org/abs/1509.02971
3. Udacity's GitHub repository **ddpg-pendulum**\
https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum
4. Udacity's jupyter notebook template of **Project: Collaboration and Competition**

