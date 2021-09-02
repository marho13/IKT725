import torch
import torch.nn as nn
import torch.nn.functional as F
# Q-value neural network class
# There is a task here!

class FCQV(nn.Module):
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
        super(FCQV, self).__init__()

        # TODO: Choose a non-linear activation function from https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        activation_fc = F.tanh # To complete. Use the format ---> F.activation_function
        self.activation_fc = activation_fc

        # TODO: propose the dimensions for the hidden layers
        hidden_dims = (input_dim * 2, input_dim * 4, input_dim * 8) # To complete. Use the format (dimension_1, ..., dimension_n)

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i]
            if i == 0:
                in_dim += output_dim
            hidden_layer = nn.Linear(in_dim, hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

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
        Forward function for neural network

        state = state from environment
        action = action from policy
        """
        x, u = self._format(state, action)
        x = self.activation_fc(self.input_layer(x))
        for i, hidden_layer in enumerate(self.hidden_layers):
            if i == 0:
                x = torch.cat((x, u), dim=1)
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x)

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