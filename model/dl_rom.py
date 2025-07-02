from math import sqrt, floor
from torch.nn import (Module, Conv2d, Flatten, Linear, Sequential, ELU,
    ConvTranspose2d)

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

class DLROM(Module):
    def __init__(
        self, 
        input_dim, 
        latent_dim, 
        in_channels, 
        out_channels, 
        hidden_channels, 
        kernels, 
        strides,
        bottleneck_hidden_dim
    ):
        super(DLROM, self).__init__()
        self.encoder = Encoder(
            input_dim, 
            latent_dim, 
            in_channels, 
            out_channels, 
            hidden_channels, 
            kernels, 
            strides, 
            bottleneck_hidden_dim
        )
        self.decoder = Decoder(
            self.encoder.out_dim, 
            self.encoder.in_dim, 
            latent_dim, 
            in_channels, 
            out_channels, 
            hidden_channels, 
            kernels, 
            self.encoder.inverse_out_padding, 
            strides,
            bottleneck_hidden_dim
        )

    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, x):
        return self.decoder(x)
        
        
class Encoder(Module):
    def __init__(self, 
        input_dim, 
        latent_dim, 
        in_channels, 
        out_channels, 
        hidden_channels, 
        kernels, 
        strides, 
        bottleneck_hidden_dim
    ):
        super(Encoder, self).__init__()

        self.in_dim = int(sqrt(input_dim))        
        self.out_dim = self.in_dim
        channels = [in_channels] + hidden_channels + [out_channels]
        conv = []
        inverse_out_padding = []
        if isinstance(strides, int):
            strides = [strides] * (len(channels) - 1)
        for i in range(len(channels) - 1):
            k, p, s = (
                kernels[i], 
                (kernels[i]-strides[i])//2 if strides[i] != 1 
                else kernels[i]//2, 
                strides[i]
            )
            conv.append(
                Conv2d(
                    channels[i], 
                    channels[i + 1], 
                    kernel_size=k,
                    stride=s, 
                    padding=p
                )
            )
            inverse_out_padding.append(self.compute_output_padding(self.out_dim, 
                conv[-1]))
            self.out_dim = self.out_dim = floor((self.out_dim + 2 * p - 
                (k - 1) - 1) / s + 1)
            conv.append(ELU())
            
            
        self.conv = Sequential(*conv)
        self.flatten = Flatten()
        self.fc = Sequential(
            Linear(
                in_features=int(out_channels * self.out_dim * self.out_dim),
                out_features=bottleneck_hidden_dim
            ),
            ELU(),
            Linear(
                in_features=bottleneck_hidden_dim,
                out_features=latent_dim
            ),
            ELU(),
        )
        self.input_channels = in_channels
        self.inverse_out_padding = inverse_out_padding
    
    @staticmethod
    def compute_output_padding(input_size, conv_layer):
        k = conv_layer.kernel_size[0]
        s = conv_layer.stride[0]
        p = conv_layer.padding[0]
        d = conv_layer.dilation[0]
    
        out = (input_size + 2 * p - d * (k - 1) - 1) // s + 1
        recon = (out - 1) * s - 2 * p + d * (k - 1) + 1
        return input_size - recon
    
    def forward(self, x):
        x = x.view(-1, self.input_channels, self.in_dim, self.in_dim)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
        
        
class Decoder(Module):
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        latent_dim, 
        in_channels, 
        out_channels, 
        hidden_channels, 
        kernels, 
        output_padding, 
        strides,
        bottleneck_hidden_dim
    ):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_channels = out_channels
        channels = [in_channels] + hidden_channels + [out_channels]
        channels = channels[::-1]
        kernels = kernels[::-1]
        output_padding = output_padding[::-1]
        strides = strides[::-1]
        conv = []
        self.dim = int(sqrt(input_dim))
        if isinstance(strides, int):
            strides = [strides] * (len(channels) - 1)
        for i in range(len(channels) - 1):
            conv.append(
                ConvTranspose2d(
                    channels[i], 
                    channels[i + 1], 
                    kernel_size=kernels[i], 
                    stride=strides[i],
                    padding=(
                        (kernels[i]-strides[i])//2 if strides[i] != 1 
                        else kernels[i]//2
                    ),
                    output_padding=output_padding[i]
                )
            )
            conv.append(ELU())
            
        self.conv = Sequential(*conv)
        self.fc = Sequential(
            Linear(
                in_features=latent_dim,
                out_features=bottleneck_hidden_dim
            ),
            ELU(),
            Linear(
                in_features=bottleneck_hidden_dim,
                out_features=int(out_channels * input_dim * input_dim)
            ),
            ELU(),
        )
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.output_channels, self.input_dim, self.input_dim)
        x = self.conv(x)
        x = x.view(-1, self.output_dim ** 2)
        return x
        
    

    