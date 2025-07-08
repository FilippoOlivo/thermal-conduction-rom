# class Normalizer:
#     def __init__(self, data):
#         self.min = data.min()
#         self.max = data.max()

#     def normalize(self, x=None):
#         return (x - self.min) / (self.max - self.min + 1e-8)

#     def unnormalize(self, x):
#         return x * (self.max - self.min + 1e-8) + self.min
        

# class NormalizerParameters:
#     def __init__(self, params):
#         self.min = params.min(0, keepdim=True).values
#         self.max = params.max(0, keepdim=True).values
    
#     def normalize(self, params):
#         return (params - self.min) / (self.max - self.min + 1e-8)
# 
import torch

class Normalizer:
    def __init__(self, data):
        self.min = data.min()
        self.max = data.max()

    def normalize(self, x=None):
        return (x - self.min) / (self.max - self.min + 1e-8)

    def unnormalize(self, x):
        return x * (self.max - self.min + 1e-8) + self.min

class NormalizerParameters:
    def __init__(self, params):
        self.min = torch.tensor([torch.min(params[..., i]) for i in range(params.shape[-1])])
        self.max = torch.tensor([torch.max(params[..., i]) for i in range(params.shape[-1])])
    
    def normalize(self, params):
        for i in range(params.shape[-1]):
            params[...,i] = (params[...,i] - self.min[i]) / (self.max[i] - self.min[i] + 1e-8)
        return params