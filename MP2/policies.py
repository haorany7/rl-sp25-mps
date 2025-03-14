from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical 

class DQNPolicy(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, device):
        super(DQNPolicy, self).__init__()
        layers = [nn.Linear(input_dim, hidden_layers[0])]
        for i, l in enumerate(hidden_layers[:-1]):
            layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.layers = nn.Sequential(*layers)
        self.device = device

    def forward(self, x):
        qvals = self.layers(x)
        return qvals 
  
    def act(self, x, epsilon=0.):
        qvals = self.forward(x)
        act = torch.argmax(qvals, 1)
        if epsilon > 0:
            act_random = torch.multinomial(torch.ones(qvals.shape[1],), 
                                           act.shape[0], replacement=True)
            act_random = act_random.to(self.device)
            combine = torch.rand(qvals.shape[0], 1) > epsilon
            combine = combine.float().to(self.device)
            act = act * combine.squeeze() + (1-combine.squeeze()) * act_random
            act = act.long()
        print("type of act inside DQNPolicy: ", type(act))
        print("shape of act inside DQNPolicy: ", act.shape)
        
        return act

class ActorCriticPolicy(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(ActorCriticPolicy, self).__init__()
        layers = [nn.Linear(input_dim, hidden_layers[0])]
        for i, l in enumerate(hidden_layers[:-1]):
            layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)
        
        self.actor = nn.Linear(hidden_layers[-1], output_dim)
        self.critic = nn.Linear(hidden_layers[-1], 1)

    def forward(self, x):
        x = self.layers(x)
        actor = self.actor(x)
        critic = self.critic(x)
        return actor, critic
  
    def actor_to_distribution(self, actor):
        action_distribution = Categorical(logits=actor.unsqueeze(-2))
        return action_distribution
        
    def act(self, x, sample=False):
        actor, critic = self.forward(x)
        action_distribution = self.actor_to_distribution(actor)
        if sample:
            action = action_distribution.sample()
            # Print before any squeezing
            # print("Original action shape:", action.shape)
            
            # Single squeeze() to ensure we get [batch_size]
            action = action.squeeze()  # Remove all singleton dimensions
            
            # print("type of action inside act: ", type(action))
            # print("shape of action inside act: ", action.shape)
            # print("val of action inside act: ", action)
        else:
            action = action_distribution.probs.argmax(-1)
            # Single squeeze() here too for consistency
            action = action.squeeze()
            
            # print("type of action inside act: ", type(action))
            # print("shape of action inside act: ", action.shape)
            # print("val of action inside act: ", action)
        return action

class QActorCriticPolicy(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(QActorCriticPolicy, self).__init__()
        
        # Initialize actor network
        actor_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            actor_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, output_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        # Initialize Q-network (critic)
        critic_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            critic_layers.append(nn.Linear(prev_dim, hidden_dim))
            critic_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        critic_layers.append(nn.Linear(prev_dim, output_dim))  # Q-values for each action
        self.critic = nn.Sequential(*critic_layers)
        
        # Initialize weights with orthogonal initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
    
    def forward(self, x):
        # Return both actor logits and Q-values
        return self.actor(x), self.critic(x)
    
    def actor_to_distribution(self, actor_logits):
        # Convert actor logits to probability distribution
        return Categorical(logits=actor_logits)
    
    def get_q_value(self, states, actions):
        # Get Q-values for specific state-action pairs
        _, q_values = self.forward(states)
        return q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    
    def act(self, states, sample=True):
        actor_logits, _ = self.forward(states)
        dist = self.actor_to_distribution(actor_logits)
        
        if sample:
            actions = dist.sample()
        else:
            actions = torch.argmax(dist.probs, dim=-1)
        
        return actions

    def get_next_value(self, x):
        """Get maximum Q-value for next state"""
        _, q_values = self.forward(x)
        return q_values.max(dim=-1)[0]


