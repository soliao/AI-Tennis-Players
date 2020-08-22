"""
This file was modified from the coding exercise of the lesson "Deep Q-Networks" of Udacity drlnd
"""

import random
import numpy as np
from collections import deque, namedtuple


## Buffer

class Buffer:
    
    """
    The replay buffer to store the experienced (state, action, reward, next_state, done) tuples
    """
    
    
    def __init__(self, buffer_size = int(1e6), batch_size = 64):
        """
        buffer_size: size of the buffer
        batch_size: size of the batch to sample for training
        """
        self.cache = deque(maxlen = buffer_size) # create a deque
        self.batch_size = batch_size
        self.replay = namedtuple("Replay", field_names = ["state", "action", "reward", "next_state", "done"])
    
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a tuple = (state, action, reward, next_state, done) into the buffer
            state: the current state
            action: the action
            reward: the reward
            next_state: the next state
            done: whether or not it reaches the terminal state
        """
        replay = self.replay(state, action, reward, next_state, done)
        self.cache.append(replay)
    
    
    def sample(self):
        """
        Sample (a batch of size self.batch_size) from the buffer
        outputs:
                states: ndarray of shape (BATCH_SIZE, 1, 48)
                actions: ndarray of shape (BATCH_SIZE, 1, 4)
                rewards: ndarray of shape (BATCH_SIZE, 2)
                next_states: ndarray of shape (BATCH_SIZE, 1, 48)
                dones: ndarray of shape (BATCH_SIZE, 2)
        """
        replays = random.sample(self.cache, k = self.batch_size)
        
        states = np.stack([x.state for x in replays if x is not None])
        actions = np.stack([x.action for x in replays if x is not None])
        rewards = np.stack([x.reward for x in replays if x is not None])
        next_states = np.stack([x.next_state for x in replays if x is not None])
        dones = np.stack([x.done for x in replays if x is not None])
        
        return (states, actions, rewards, next_states, dones)

    
    def __len__(self):
        """
        Returns the number of replays stored in the buffer
        """
        return len(self.cache)