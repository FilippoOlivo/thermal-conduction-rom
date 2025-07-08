from torch.nn import Module, Linear, Sequential, ELU

class InterpolationNetwork(Module):
    def __init__(self, input_dim, latent_dim, layers):
        super(InterpolationNetwork, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        layers = [input_dim] + layers + [latent_dim]
        
        network = []
        for i in range(len(layers) - 1):
            network.append(
                Linear(
                    in_features=layers[i],
                    out_features=layers[i + 1]
                )
            )
            if i < len(layers) - 2:
                network.append(ELU())
        self.network = Sequential(*network)
    
    def forward(self, x):
        return self.network(x)