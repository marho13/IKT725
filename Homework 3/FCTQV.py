import torch
import torch.nn as nn
import torch.nn.functional as F
# Q-value neural network class - Double Q-learning
# There is a task here!

class FCTQV(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim):
        """
        Class initialization

        input_dim = input dimension
        output_dim = output dimension

        hidden_dims = dimension for hidden layers
        activation_fc = activation function

        device = processing device
        """
        super(FCTQV, self).__init__()

        # TODO: Choose a non-linear activation function from https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        activation_fc =  F.tanh# To complete. Use the format ---> F.activation_function
        self.activation_fc = activation_fc

        # TODO: propose the dimensions for the hidden layers
        hidden_dims = (input_dim*2, input_dim*4, input_dim*8)# To complete. Use the same format (dimension_1, ..., dimension_n)

        # Initialize Q-value neural network A
        self.input_layer_a = nn.Linear(input_dim + output_dim, hidden_dims[0])
        self.hidden_layers_a = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            hidden_layer_a = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers_a.append(hidden_layer_a)

        self.output_layer_a = nn.Linear(hidden_dims[-1], 1)

        # TODO: initialize the second Q-value neural network - Use the format nn_property_b
        # ---YOUR CODE GOES HERE---

        # -------------------------

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

    def _format(self, state, action):
        """
        Format the state for pytorch

        state = state from environment
        action = action from policy
        """
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             device=self.device,
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u,
                             device=self.device,
                             dtype=torch.float32)
            u = u.unsqueeze(0)
        return x, u

    def forward(self, state, action):
        """
        Forward function for neural network - Q-value

        state = state from environment
        action = action from policy
        """
        x, u = self._format(state, action)
        x = torch.cat((x, u), dim=1)

        xa = self.activation_fc(self.input_layer_a(x))
        xb = self.activation_fc(self.input_layer_b(x))

        for hidden_layer_a, hidden_layer_b in zip(self.hidden_layers_a, self.hidden_layers_b):
            xa = self.activation_fc(hidden_layer_a(xa))
            xb = self.activation_fc(hidden_layer_b(xb))

        xa = self.output_layer_a(xa)
        xb = self.output_layer_b(xb)
        return xa, xb

    def Qa(self, state, action):
        """
        Forward function for neural network - policy

        state = state from environment
        action = action from policy
        """
        x, u = self._format(state, action)
        x = torch.cat((x, u), dim=1)
        xa = self.activation_fc(self.input_layer_a(x))
        for hidden_layer_a in self.hidden_layers_a:
            xa = self.activation_fc(hidden_layer_a(xa))
        return self.output_layer_a(xa)

    def load(self, experiences):
        """
        load samples from experience - replay buffer database

        experiences = samples from the replay buffer database
        """
        states, actions, new_states, rewards, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, new_states, rewards, is_terminals