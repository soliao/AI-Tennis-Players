# Report.md


## Multi-Agent Deep Deterministic Policy Gradient (MADDPG)



### MADDPG paper

R. Lowe et al., 2017. *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments* (https://arxiv.org/abs/1706.02275)


### The basics of MADDPG
A brief introduction of the Deep Deterministic Policy Gradient (DDPG) can be found in my previous project (https://github.com/sliao-mi-luku/DeepRL-continuous-control-reachers-udacity-drlnd-p2). The Multi-Agent Deep Deterministic Policy Gradient (MADDPG) extends the basic concepts of DDPG and can be better applied in an environment where there are mutiple agents competing or cooperating with each other. Similar to the structure of DDPG, in MADDPG each agent has an *actor* and a *critic*:

**The actor**

Each agent has its own policy (called the actor) to interact with the environment. The policy only considers the states that can be observed by the agent. The actor can not access to the states observed by other agents.

**The critic**

In addition to the actor, each agent also has its own critic to evaluate and improve its policy. We can view the critic as the **personal coach** of the agent. The critic can see everything in the environment: not only the information observed by all agents, but also the actions taken by all agents. The critic then uses all the information it has to evaluate the action-value of the policy of the agent which it's responsible for. By assigning every agent with a unique critic, we can generalize the application of MADDPG in both competitive or coorperative environment because the critics work independently of each other and is only dedicated to the success of the agent it's coaching.

For each agent, the actor and critic are deep neural networks consisting of some fully connected layers. Similar to the DQN implementation, the actor has a *local* and *target* networks (and so does the critic). The target network is soft-updated at the end of each round of batch-training.

In this project there're 2 agents playing tennis with each other, and the goal is to hit the ball over the net as many times as possible.


### Neural network architectures

**The actor**

The actor network learns the policy of the agent. Each agent has its own actor network. The input to the actor network
is the state vector which is only observed by the agent (size = 24), and the output will be the action taken by the agent (size = 2). Each component of the action 
vector takes a continuous real value in the range [-1, 1].

The default architecture of the actor (both local and target) network consists of:

- **input layer**: a fully connected layer with input size = 24, output size = 400, activation function = ReLU
- **hidden layer**: a fully connected layer with input size = 400, output size = 300, activation function = ReLU
- **output layer**: a fully connected layer with input size = 300, output size = 2, activation function = tanh


**The critic**

The critic network learns how good/bad the policy is. Each agent has its own critic network. The critic network takes
the **states observed by both agents** and the **actions taken by both agents** as the inputs, and outputs an action-value **q(s, a)**

The default architecture of the critic (both local and target) network consists of:

- **input layer**: a fully connected layer with input size = 48, output size = 400, activation function = ReLU
- **hidden layer**: a fully connected layer with input size = 404, output size = 300, activation function = ReLU
- **output layer**: a fully connected layer with input size = 300, output size = 1

**Notes** I used the architecture used in the DDPG paper (https://arxiv.org/abs/1509.02971), the state vector (size = 48) is passed through the input layer alone.
After the ReLU activation, the output (size = 400) is concatenated with the action vector (size = 4) and becomes the input (size = 404) to the next hidden layer.

More details can be found in the file `networkModels.py`


### The noise process
To motivate exploration, once the actor determines the action vector, the agent will add some noise to it and use the new vector as the action to interact with the environment. Adding a larger noise means the agent will explore farther away from its current policy (exploration), and adding a smaller noise means the agent will stay closer to its cuurent policy (exploitation).

Similar to the **DDPG paper**, I modeled the noise by the **Ornstein-Uhlenbeck Process** (OU noise).

`OUnoise(t) = OUnoise(t-1) + ou_theta*(asymptotic_mean - OUnoise(t-1)) + ou_sigma*Gaussian_diffusion`

The OU noise has 2 important parameters:

`ou_theta` controls the magnitude of the drift term `asymptotic_mean - OUnoise(t-1)` (default = 0.15)

`ou_sigma` controls the magnitude of the diffusion term (default = 0.20)


As seen in the equation, the direction of the drift term is always pointing toward the asymptotic mean (which is set to be 0) of the noise to avoid divergence.  

To motivate exploration in the beginning of the training and motivate exploitation as the training goes on, I multiplied the noise with a scaling factor `ou_scale` which decays exponentially. In the first episode, `ou_scale` is 1.0. After each episode, `ou_scale` is multiplied by a constant `ou_decay` (0.9995).

For more details of the Ornstein-Uhlenbeck Process, please refer to the textbook *Stochastic Methods, a handbook for the natural and social sciences* 
by Crispin Gardiner (https://www.springer.com/gp/book/9783540707127).


### The MADDPG algorithm

Below is a brief description of MADDPG:

- initialize the 2 agents with the actor (local and target) and critic (local and target) networks
- t_step = 0
- each agent i observes the initial state **s_i** (size = 2*24 = 48) from the environment
- while the environment is not solved:
  - t_step += 1
  - each agent i uses *actor_local* to determine the action from its own observation: `**a_i** = actor_local(s_i)`
  - each agent i uses **a_i** to interact with the environment
  - each agent i collects the reward **r_i** and enters the next state (**s'_i**)
  - combine the **s_i**, **a_i**, **r_i**, and **s'_i** of each agent i into the vectors **s**, **a**, **r**, and **s'** respectively
  - add the experience tuple **(s, a, r, s')** into the replay buffer
  
  - if there are enough (>= `batch_size`) replays in the buffer:
    - loop for for each agent i:
    
        **step 1: sample replays**
        
        1-1 randomly sample a batch of size = `batch_size` replay tuples **(s, a, r, s')** from the buffer

        **step 2: train `critic_local`**
        
        2-1 use *actor_target* predict the future actions **a'** = actor_target(s')
        
        2-2 use *critic_target* to predict the action value of the next state **q_next** = critic_target(s', a')
        
        2-3 calculate the TD target of the action value **q_target** = **r** + gamma * **q_next**
        
        2-4 use *critic_local* to calculate the current action value **q_local** = critic_local(s, a)
        
        2-5 define the loss function (Huber loss) to be the TD error, i.e.,\
        `critic_loss = F.smooth_l1_loss(q_local, q_target)`
        
        2-6 use gradient decent to update the weights of *critic_local*

        **step 3: train `actor_local`**
        
        3-1 use *actor_local* to determine the action **a_local** = actor_local(s)
        
        3-2 use *critic_local* to calculate the action values and averaged over the sample batch to obtain the loss of actor:\
        `actor_loss = -critic_local(s, a_local).mean()` (averaged over the batch)
        
        3-3 use gradient descent to update the weights of *actor_local*

        **step 4: update `actor_target` and `critic_target`**
        
        4-1 soft-update the weights of *actor_target* and *critic_target*\
        `actor_target.weights <-- tau * actor_local.weights + (1-tau) * actor_target.weights`\
        `critic_target.weights <-- tau * critic_local.weights + (1-tau) * critic_target.weights`
  - **s** <-- **s'**


## Hyperparameters

| Hyperparameter | Value | Description | Reference |
| ----------- | ----------- | ----------- | ----------- |
| actor_hidden_sizes | [400, 300] | sizes of actor's hidden FC layers | <1> |
| critic_hidden_sizes | [400, 300] | sizes of critic's the hidden FC layers | <1> |
| gamma | 0.99 | discount rate | <1> |
| actor_lr | 1e-4 | actor's learning rate | <1> |
| critic_lr | 1e-3 | critic's learning rate | <1> |
| critic_L2_decay | 0 | critic's L2 weight decay |  |
| tau | 1e-3 | soft update | <1> |
| ou_scale | 1.0 | initial scaling of noise |  |
| ou_decay | 0.9995 | scaling decay of noise |  |
| ou_mu | 0.0 | initial mean of noise | <1> |
| ou_theta | 0.15 | drift component of noise | <1> |
| ou_sigma | 0.20 | diffusion component of noise | <1> |
| buffer_size | 1e5 | size of the replay buffer |  |
| batch_size | 128 | size of the minibatch |  |

<1> Theses are the values used in the original DDPG paper (https://arxiv.org/pdf/1509.02971.pdf)
Please refer to the **Experiment details** in the paper for more information.


## Training result

With the parameters above, the agent solved the task after 841 episodes, i.e., the average (over agents) score from episode #842 to #941 reaches above +0.5 points.

[![p3-scores.png](https://i.postimg.cc/HLwxpBw6/p3-scores.png)](https://postimg.cc/rdpcjGY4)\
**(figure)** *Average score of each episode*


## Ideas for future work

In the MADDPG framework, the information **observed by both players** are used to trained the critic networks. In this tennis project, the states observed by each agent include the kinematics (position and velocity) of the ball. In other words, the state vectors may have some redundant terms that are *identical* or *with the same absolute value but in opposite signs*. My current network model is equivalent to telling the critic about the location of the ball twice - one in Agent 0's coordinate and one in Agent 1's coordinate. Why not just telling the critic the ball's coordinate in a single reference coordinate?

Therefore, I think the size of the state vector can be reduced in this environment, as long as the exact meaning of each component of the state vector is completely known. My future plan is to look closely at the trajectories of the states from the replays and to figure out which terms are highly correlated to each other. By reducing the redundancy, training the actor and critic networks can be more efficient.

## References in this project
1. R. Lowe et al., 2017. *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*\
https://arxiv.org/abs/1706.02275
2. T. P. Lillicrap et al., 2016. *Continuous control with deep reinforcement learning*\
https://arxiv.org/abs/1509.02971
3. Udacity's GitHub repository **ddpg-pendulum**\
https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum
4. Udacity's drlnd jupyter notebook template of **Project: Collaboration and Competition**
5. Udacity's drlnd MADDPG-Lab (maddpg.py)
