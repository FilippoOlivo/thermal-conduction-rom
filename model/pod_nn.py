import torch
from pina.model.block import PODBlock
from pina.model import FeedForward

class PODNN(torch.nn.Module):
    def __init__(self, pod_rank, layers, func=torch.nn.Softplus):
        super().__init__()
        self.pod = PODBlock(pod_rank, scale_coefficients=False)
        self.nn = FeedForward(
            input_dimensions=3,
            output_dimensions=pod_rank,
            layers=layers,
            func=func,
        )
        

    def forward(self, p):
        coefficients = self.nn(p)
        return self.pod.expand(coefficients)

    def fit_pod(self, x):
        self.pod.fit(x)