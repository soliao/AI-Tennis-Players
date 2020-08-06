# RL_MADDPG_Tennis_Unity-ML-Agents_Udacity-drlnd-env
Unity Machine Learning Agents (ML-Agents) Tennis Environment

The environment of the project is by Udacity's Deep Reinforcement Learning Nanodegree (drlnd), which is similar to ML-Agents but not identical

### Project Details

In the environment, two agents (with blue and red rackets) are boucing a ball with each other. The goal is to train both agents so that they
make as more rallies as possible.

In reinfoecement learning, we use states, actions and rewards to define the problem:

**states**\
For each agent, the state space represents:\

**actions**\
For each agent, the action space has 2 continuous components ax, ay

**rewards**\
During each episode, each agent receives a reward of **+0.1** if it hits the ball over the net, and receives a reward of **-0.01** if it
doesn't hit the ball back or if it hits the ball out of the court. The agent collects the reward without discounting

**solving condition**\
According to Udacity DRLND, the environment is considered solved if the average score S (defined below) over 100 consecutive episodes is
larger than +0.5

score S of an episode is defined as **the maximum reward of the 2 agents in that episode**

### Getting Started




### Instructions (for Windows user)

1. Open **SL_Tennis_MADDPG** with Jupyter
  * use the command jupyter
2. Run Box 1 to import necessary packages
  * Paste the path to **.../Tennis_Windows_x86_64/Tennis.exe** into the command ***env = UnityEnvironment(file_name = "...")***
3. 
