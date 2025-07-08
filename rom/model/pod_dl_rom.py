import torch
from torch.nn import Module
from pina.model.block import PODBlock
from .dl_rom import DLROM

class POD_DL_ROM(Module):
    def __init__(self, 
        input_dim,
        pod_modes, 
        input_channels,
        output_channels,
        latent_dim, 
        hidden_channels, 
        kernels, 
        strides,
        bottleneck_hidden_dim
    ):
        super(POD_DL_ROM, self).__init__()
        self.pod_modes = pod_modes
        self.pod = PODBlock(
            pod_modes, 
            scale_coefficients=False, 
        )
        self.dl_rom = DLROM(
            pod_modes, 
            latent_dim, 
            input_channels, 
            output_channels, 
            hidden_channels, 
            kernels, 
            strides, 
            bottleneck_hidden_dim
        )
        self.register_buffer('basis', torch.empty(pod_modes, input_dim))
    
    def fit(self, data):
        """
        Fit the POD model to the data.
        :param data: The input data for training.
        """
        self.pod.fit(data)
        self.basis = self.pod.basis
    
    
    def encode(self, x):
        """
        Decode the data using the POD model.
        :param data: The input data to decode.
        :return: The decoded data.
        """
        x = x @ self.basis.T
        return self.dl_rom.encode(x)
    
    def decode(self, x):
        """
        Decode the data using the POD model.
        :param data: The input data to decode.
        :return: The decoded data.
        """
        x = self.dl_rom.decode(x)
        x = x @ self.basis
        return x
        