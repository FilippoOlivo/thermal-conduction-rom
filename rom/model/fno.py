from pina.model import FNO
import torch

class FNO2d(FNO):
    def __init__(
        self,
        input_channels,
        output_channels,
        n_modes,
        padding,
        inner_size,
        n_layers,
        func=torch.nn.Tanh,
    ):
        lift_net = torch.nn.Sequential(
            torch.nn.Linear(input_channels, inner_size // 2),
            func(),
            torch.nn.Linear(inner_size // 2, inner_size),
            func(),
        )
        
        project_net = torch.nn.Sequential(
            torch.nn.Linear(inner_size, inner_size // 2),
            func(),
            torch.nn.Linear(inner_size // 2, output_channels),
        )
        
        super().__init__(
            lifting_net=lift_net,
            projecting_net=project_net,
            n_modes=n_modes,
            dimensions=2,
            padding=padding,
            padding_type="constant",
            inner_size=inner_size,
            n_layers=n_layers,
            func=func,
        )
        
        