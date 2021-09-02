import torch
import torch.nn as nn
import torch.nn.functional as F
# Deterministic policy neural network class
# There is a task here!

class FCDP(nn.Module):
    def __init__(self,
                 input_dim,
                 action_bounds,
                 out_activation_fc=F.tanh):
        """
        Class initialization

        input_dim = input dimension
        output_dim = output dimension

        hidden_dims = dimension for hidden layers
        activation_fc = activation function
        out_activation_fc = Output activation function

        device = processing device
        """
        super(FCDP, self).__init__()

        # TODO: Choose a non-linear activation function from https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        activation_fc =  F.tanh# To complete. Use the format ---> F.activation_function

        self.activation_fc = activation_fc
        self.out_activation_fc = out_activation_fc
        self.env_min, self.env_max = action_bounds

        # TODO: propose the dimensions for the hidden layers
        hidden_dims = (input_dim * 2, input_dim * 4, input_dim * 8)# To complete. Use the same format (dimension_1, ..., dimension_n)

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], len(self.env_max))

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

        self.env_min = torch.tensor(self.env_min,
                                    device=self.device,
                                    dtype=torch.float32)

        self.env_max = torch.tensor(self.env_max,
                                    device=self.device,
                                    dtype=torch.float32)

        self.nn_min = self.out_activation_fc(
            torch.Tensor([float('-inf')])).to(self.device)
        self.nn_max = self.out_activation_fc(
            torch.Tensor([float('inf')])).to(self.device)
        self.rescale_fn = lambda x: (x - self.nn_min) * (self.env_max - self.env_min) / \
                                    (self.nn_max - self.nn_min) + self.env_min

    def _format(self, state):
        """
        Format the state for pytorch

        state = current state
        """
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             device=self.device,
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        """
        Forward function for neural network

        state = current state
        """
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        x = self.out_activation_fc(x)
        return self.rescale_fn(x)