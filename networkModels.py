"""
Reference:
This file was modified from model.py from Udacity's GitHub Repository ddpg-pendulum
https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

## Actor Neural Network

class ActorNetwork(nn.Module):
    """
    The actor network outputs the action of a single agent by using the state (observed by the same agent) as the input
    In this project, each agent (player) observes the state of dimension = 24 and has the action (continuous value) of dimension = 2
    The values of the action are continuous, ranging from -1 to +1
    """
    
    def __init__(self, input_size = 24, output_size = 2, hidden_sizes = [400, 300]):
        """
        input_size: the dimension of the state vector of a single agent
        output_size: the dimension of the action vector of a single agent
        hidden_sizes: the sizes of the input and output units of the hidden layer
                      for example, hidden_sizes = [400, 300] means the hidden layer has input_size = 400, and output_size = 300
        """
        super(ActorNetwork, self).__init__()

        self.hidden_layers = nn.ModuleList([])
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0])) # the first layer (the input layer)

        for h1, h2 in zip(hidden_sizes[:-1], hidden_sizes[1:]): # from the sencond layer to the (last-1) layer
            self.hidden_layers.append(nn.Linear(h1, h2))
            
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size) # the last layer (the output layer)
        
        self.reset_parameters() # reset the initial weights and biases of all layers
    
    def reset_parameters(self):
        """
        The heuristic method for choosing the initial scales of the weights and biases of a fully connected layer
        """
        for layer in self.hidden_layers:
            # fan_in
            f = layer.weight.data.size()[0]
            layer.weight.data.uniform_(-1.0/np.sqrt(f), 1.0/np.sqrt(f))
            layer.bias.data.fill_(0.1)
            
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)
        self.output_layer.bias.data.fill_(0.1)
        
    def forward(self, x):
        """
        Forward pass of the state tensor through the network.
        The network returns the action tensor, with the value of each dimension ranging from -1 to +1
        
        x: the input tensor (the state)        
        """
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return F.tanh(self.output_layer(x))


## Critic Neural Network

class CriticNetwork(nn.Module):
    """
    The critic network
    
    Given the states and actions of "both agents", the critic network outputs the action-value Q(s1, s2, a1, a2)
    """
    
    def __init__(self, state_size = 48, action_size = 4, hidden_sizes = [400, 300]):
        """
        state_size: the dimension of the state vector of both agents (2*24 = 48)
        action_size: the dimension of the action vector of both agents (2*2 = 4)
        hidden_sizes: the sizes of the input and output units of the hidden layer (the layer after the input layer)
                      for example, hidden_sizes = [400, 300] means the hidden layer has input_size = 400, and output_size = 300
        """
        super(CriticNetwork, self).__init__()
        
        self.first_layer = nn.Linear(state_size, hidden_sizes[0]) # the first layer (the input layer)
        self.second_layer = nn.Linear(hidden_sizes[0] + action_size, hidden_sizes[1]) # the second layer (the hidden layer)
        self.output_layer = nn.Linear(hidden_sizes[1], 1) # the last layer (the output layer)
        
        self.reset_parameters() # reset the initial weights and biases of all layers
    
    def reset_parameters(self):
        """
        The heuristic method for choosing the initial scales of the weights and biases of a fully connected layer
        """
        f1 = self.first_layer.weight.data.size()[0]
        f2 = self.second_layer.weight.data.size()[0]

        self.first_layer.weight.data.uniform_(-1.0/np.sqrt(f1), 1.0/np.sqrt(f1))
        self.second_layer.weight.data.uniform_(-1.0/np.sqrt(f2), 1.0/np.sqrt(f2))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)
        
        self.first_layer.bias.data.fill_(0.1)
        self.second_layer.bias.data.fill_(0.1)
        self.output_layer.bias.data.fill_(0.1)
        
    def forward(self, state, action):
        """
        Forward pass of the state and action tensors through the network.
        The network returns the action-value Q(s, a)
        
        state: the state tensor (dim = 48)
        action: the action tensor (dim = 4)
        """
        xs = F.relu(self.first_layer(state)) # only pass the state tensor through the first FC layer
        x = torch.cat((xs, action), dim = 1) # concatenate with the raw action tensor
        x = F.relu(self.second_layer(x)) # pass through the second FC layer together
        x = self.output_layer(x) # pass through the output FC layer
        return x