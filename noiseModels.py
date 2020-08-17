"""
Reference:
This file was modified from ddpg_agent.py from Udacity's GitHub Repository ddpg-pendulum
https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py
"""

## Ornstein–Uhlenbeck Process

import numpy as np
import matplotlib.pyplot as plt


class OUNoise:
    
    """
    The Ornstein-Uhlenbeck process generating the noise for exploration
    """
    
    def __init__(self, output_size = 4, mu = 0.0, theta = 0.15, sigma = 0.20):
        
        """
        output_size: dimension of the noise, which should be the dimension of the action
        mu: the asymptotic mean of the noise
        theta: the magnitude of the drift component
        sigma: the magnitude of the diffusion (Gaussian noise) component
        """
               
        self.output_size = output_size
        self.mu = mu*np.ones(output_size)
        self.theta = theta
        self.sigma = sigma

        self.reset()

    def reset(self):
        
        """
        Set the current noise value to the asymptotic mean value
        """
        
        self.x = np.copy(self.mu)

    def get_noise(self):
        
        """
        Generate a noise vector of dimension = self.output_size
        """
        
        self.x += self.theta*(self.mu - self.x) + self.sigma*np.random.randn(self.output_size)
        return self.x
    
    
def plot_OU(t_max = 1000, A0 = 1, decay_factor = 1, output_size = 1, mu = 0.0, theta = 0.15, sigma = 0.20):
    
    """
    This is a helper function to view the noise process by simulating t_max time steps
    
    t_max: maximum time step for simulation
    A0: the initial scaling factor of the noise
    decay_factor: the decay rate of the scaling factor after each time step
    output_size: dimension of the noise, which should be the dimension of the action
    mu: the asymptotic mean of the noise
    theta: the magnitude of the drift component
    sigma: the magnitude of the diffusion (Gaussian noise) component
    """
    
    OU = OUNoise(output_size, mu, theta, sigma) # Ornstein–Uhlenbeck Process
    t = list(range(t_max))
    x = []
    A = A0 # scaling factor
    for _ in t:
        x.append(np.mean(A*OU.get_noise())) # scaling factor * the noise value
        A *= decay_factor # decrease the scaling factor by decay rate = decay_factor
    plt.figure()
    plt.plot(t, x)
    plt.title('Simulated noise process')
    plt.xlabel('Episodes')
    plt.ylabel('Noise')
    plt.show()