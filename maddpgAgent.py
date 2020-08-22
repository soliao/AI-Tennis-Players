"""
Reference:
This file was modified from ddpg_agent.py from Udacity's GitHub Repository ddpg-pendulum
https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py
"""

import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from networkModels import ActorNetwork, CriticNetwork


## MADDPG_Agent

class MADDPG_Agent:
    
    """
    An MADDPG_Agent includes:
        2 actor (actor_local, actor_target) networks
        2 critic (critic_local, critic_target) networks
    """
    
    def __init__(self, agent_index, config):
        
        """
        agent_index: index of the agent (0 or 1)
        
        config: the dictionary containing the keys:
            actor_input_size: input size of the actor (24, dimension of the state of a single agent)
            actor_output_size: output size of the actor (2, dimension of the action of a single agent)
            actor_hidden_sizes: input and output sizes of the hidden FC layer of the actor
            
            critic_state_size: sum of the dimensions of the state of both agents (48)
            critic_action_size: sum of the dimensions of the action of both agents (4)
            critic_hidden_sizes: input and output sizes of the hidden FC layer of the critic
            
            actor_lr: learning rate of the actor
            critic_lr: learning rate of the critic
            critic_L2_decay: L2 weight decay of the critic
            
            gamma: the discounting rate
            
            tau: soft-update factor
        """
        
        self.agent_index = agent_index
        
        actor_input_size = config['actor_input_size']
        actor_output_size = config['actor_output_size']
        actor_hidden_sizes = config['actor_hidden_sizes']
        
        critic_state_size = config['critic_state_size']
        critic_action_size = config['critic_action_size']
        critic_hidden_sizes = config['critic_hidden_sizes']
        
        actor_lr = config['actor_lr']
        critic_lr = config['critic_lr']
        critic_L2_decay = config['critic_L2_decay']
        
        self.gamma = config['gamma']
        
        self.tau = config['tau']
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        
        ## Actor networks (local & target)
        self.actor_local = ActorNetwork(actor_input_size, actor_output_size, actor_hidden_sizes).to(self.device)
        self.actor_target = ActorNetwork(actor_input_size, actor_output_size, actor_hidden_sizes).to(self.device)
        
        # set actor_local and actor_target with same weights & biases
        for local_param, target_param in zip(self.actor_local.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(local_param.data)
        
        ## Critic networks (local & target)
        self.critic_local = CriticNetwork(critic_state_size, critic_action_size, critic_hidden_sizes).to(self.device)
        self.critic_target = CriticNetwork(critic_state_size, critic_action_size, critic_hidden_sizes).to(self.device)
        
        # set critic_local and critic_target with same weights & biases
        for local_param, target_param in zip(self.critic_local.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(local_param.data)
        
        # optimizers
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = critic_lr, weight_decay = critic_L2_decay)
        
        
    def act(self, state_t, actor_name = 'target', noise_bool = False, noise_np = None):
        """
        Use the actor network to determine the action
            
        inputs:
            state_t: state tensor of shape (m, 24) observed by the agent
            actor_name: the actor network to use ("local" or "target")
            noise_bool: whether or not to add the noise
            noise_np - the noise to be added (if noise_bool == True), an ndarray of shape (m, 2)
        output:
            the action (tensor) of shape (m, 2)
        """
        if actor_name == 'local':
            actor_network = self.actor_local
        elif actor_name == 'target':
            actor_network = self.actor_target
        
        actor_network.eval()
        with torch.no_grad():
            action = actor_network(state_t).float().detach().to(self.device) # action is a tensor
            
            if noise_bool: # to add noise
                action = action.cpu().data.numpy() # convert action to ndarray
                action = np.clip(action + noise_np, -1, 1) # add noise and clip between [-1, +1]
                action = torch.from_numpy(action).float().detach().to(self.device) # convert action to tensor
        actor_network.train()
        
        return action       
    
    
    def soft_update(self, local_nn, target_nn):
        """
        Soft-update the weight of the actor (or critic) target network
        """
        for local_param, target_param in zip(local_nn.parameters(), target_nn.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
            
    
    def learn(self, replays, other_agent):
        """        
        Used the sampled replays to train the actor and the critic
        
        replays: replay tuples in the format of (states, actions, rewards, next_states, dones)
            
            states.shape = (m, 48)
            actions.shape = (m, 4)
            rewards.shape = (m, 2)
            next_states.shape = (m, 48)
            dones.shape = (m, 2)
        
        other_actor: the other agent
        """
                        
        ## assign s, a, r, s', d from replays
        states, actions, rewards, next_states, dones = replays

        # size of the batch
        m = actions.shape[0]
        
        ## convert from ndarrays to tensors
        
        if self.agent_index == 0:
            # states, actions, next_actions of the agent
            states_self = torch.from_numpy(states[:, :24]).float().to(self.device)             # [m, 24]
            action_self = torch.from_numpy(actions[:, :2]).float().to(self.device)             # [m, 2]
            next_states_self = torch.from_numpy(next_states[:, :24]).float().to(self.device)   # [m, 24]
            # states, actions, next_actions of the other agent
            states_other = torch.from_numpy(states[:, 24:]).float().to(self.device)            # [m, 24]
            action_other = torch.from_numpy(actions[:, 2:]).float().to(self.device)            # [m, 2]
            next_states_other = torch.from_numpy(next_states[:, 24:]).float().to(self.device)  # [m, 24]
            # rewards and dones
            rewards = torch.from_numpy(rewards[:, 0].reshape((-1, 1))).float().to(self.device)                  # [m, 1]
            dones = torch.from_numpy(dones[:, 0].reshape((-1, 1)).astype(np.uint8)).float().to(self.device)     # [m, 1]
            
        elif self.agent_index == 1:
            # states, actions, next_actions of the agent
            states_self = torch.from_numpy(states[:, 24:]).float().to(self.device)             # [m, 24]
            action_self = torch.from_numpy(actions[:, 2:]).float().to(self.device)             # [m, 2]
            next_states_self = torch.from_numpy(next_states[:, 24:]).float().to(self.device)   # [m, 24]
            # states, actions, next_actions of the other agent
            states_other = torch.from_numpy(states[:, :24]).float().to(self.device)            # [m, 24]
            action_other = torch.from_numpy(actions[:, :2]).float().to(self.device)            # [m, 2]
            next_states_other = torch.from_numpy(next_states[:, :24]).float().to(self.device)  # [m, 24]
            # rewards and dones
            rewards = torch.from_numpy(rewards[:, 1].reshape((-1, 1))).float().to(self.device)                  # [m, 1]
            dones = torch.from_numpy(dones[:, 1].reshape((-1, 1)).astype(np.uint8)).float().to(self.device)     # [m, 1]
        
        # s, a, s' for both agents
        states = torch.from_numpy(states).float().to(self.device)                 # [m, 48]
        actions = torch.from_numpy(actions).float().to(self.device)               # [m, 4]
        next_states = torch.from_numpy(next_states).float().to(self.device)       # [m, 48]
                
        
        """ Train critic_local """
        # next_actions of the agent
        next_actions_self = self.act(state_t = next_states_self, actor_name = 'target', noise_bool = False)
        
        # next_actions of the other
        next_actions_other = other_agent.act(state_t = next_states_other, actor_name = 'target', noise_bool = False)
                
        # combine next actions from both agents
        if self.agent_index == 0:
            next_actions = torch.cat([next_actions_self, next_actions_other], dim = 1).float().detach().to(self.device) # (m, 4)
        elif self.agent_index == 1:
            next_actions = torch.cat([next_actions_other, next_actions_self], dim = 1).float().detach().to(self.device) # (m, 4)
         
        
        # q_next: use critic_target to obatin the action-value of (next_states, next_actions)
        self.critic_target.eval()
        with torch.no_grad():
            q_next = self.critic_target(next_states, next_actions).detach().to(self.device) # [m, 1]
        self.critic_target.train()
        
        
        # q_target: the TD target of the critic, i.e. q_target = r + gamma*q_next
        q_target = rewards + self.gamma * q_next * (1-dones) # [m, 1]
        
        # q_local: the current action-value of (states, actions)
        q_local = self.critic_local(states, actions) # [m, 1]
        
        # critic_loss
        self.critic_optimizer.zero_grad()
        critic_loss = F.smooth_l1_loss(q_local, q_target.detach())
        critic_loss.backward()
        self.critic_optimizer.step()

        
        """ Train actor_local """
        # action_local
        if self.agent_index == 0:
            action_local = torch.cat([self.actor_local(states_self),
                                      other_agent.act(states_other, actor_name = 'local', noise_bool = False)],
                                      dim = 1)
        elif self.agent_index == 1:
            action_local = torch.cat([other_agent.act(states_other, actor_name = 'local', noise_bool = False),
                                      self.actor_local(states_self)],
                                      dim = 1)
                       
        # actor_loss
        self.actor_optimizer.zero_grad()
        actor_loss = - self.critic_local(states, action_local).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        ## soft-update actor_target and critic_target
        self.soft_update(self.actor_local, self.actor_target)
        self.soft_update(self.critic_local, self.critic_target)