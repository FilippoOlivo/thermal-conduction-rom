import torch
from torch.nn import Module
from pina.model.block import PODBlock
from model.dl_rom import DLROM

class POD_DL_ROM(Module):
    def __init__(self, 
            pod_modes, 
            input_channels,
            output_channels,
            input_dim, 
            latent_dim, 
            hidden_channels, 
            kernels, 
            strides
    ):
        super(POD_DL_ROM, self).__init__()
        self.pod_modes = pod_modes
        self.pod_block = PODBlock(pod_modes)
        self.dl_rom = DLROM(pod_modes, latent_dim, input_channels, output_channels, hidden_channels, kernels, strides)
        self.register_buffer('basis', torch.empty(1,1))
    
    def fit(self, data):
        """
        Fit the POD model to the data.
        :param data: The input data for training.
        """
        self.pod_block.fit(data)
        print(self.pod_block.basis.shape)
        self.basis = self.pod_block.basis.clone()
    
    
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
        