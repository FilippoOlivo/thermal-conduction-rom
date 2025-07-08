import torch 
import numpy as np
from pina.model import FNO
from pina.problem.zoo import SupervisedProblem
from pina.solver import SupervisedSolver
from pina.optim import TorchOptimizer
from pina.trainer import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import h5py
from tqdm import tqdm
from matplotlib import pyplot as plt

params_ = np.load("data/data_vert_raw.npz")['parameters']
simulations_ = np.load("data/data_vert_raw.npz")['simulations']


with h5py.File('solution.h5', 'r') as f:
    x_, y_ = f['x'][:], f['y'][:]  
    
x = np.sort(np.unique(x_))
y = np.sort(np.unique(y_))

params = np.zeros((len(params_), len(x), len(y), 4))
simulations = np.zeros((len(params_), len(x), len(y), 1))

map = np.zeros((len(x_), len(y_)), dtype=int)
for i in range(len(x)):
    idx_x = np.where(x_ == x[i])[0]
    for j in range(len(y)):
        idx_y = np.where(y_ == y[j])[0]
        map[j, i] = np.intersect1d(idx_x, idx_y)[0]

for i in tqdm(range(len(params_))):
    for j in range(len(x)):
        for k in range(len(y)):
            params[i, k, j, 0] = x[j]
            params[i, k, j, 1] = y[k]
            if x[j] < 1e-6:
                params[i, k, j, 2] = params_[i, 2]
            if y[k] <= params_[i, 0]:
                params[i, k, j, 3] = 20
            elif y[k] <= params_[i, 1]:
                params[i, k, j, 3] = 10
            else:
                params[i, k, j, 3] = 1
            
            simulations[i, k, j, 0] = simulations_[i, map[k, j]]
            
plt.pcolor(x,y,simulations[0, :, :, 0], cmap='viridis')
plt.colorbar()
plt.savefig("simulations_example.png")
plt.close()
plt.pcolor(x,y,params[0, :, :, 3], cmap='viridis')
plt.colorbar()
plt.savefig("params_example.png")
points = params[0, :, : , [0,1]]
points = points.reshape(points.shape[0], points.shape[1] * points.shape[2]).T
print(points.shape)
np.savez("data/data_vert_no.npz", simulations=simulations, parameters=params, points=points)
# simulations = simulations.reshape(simulations.shape[0], simulations.shape[1] ** 2)
# params = params.reshape(params.shape[0], params.shape[1] ** 2, params.shape[-1])

# print(points.shape)
# print("Points:", points[:,0].max(), points[:,1].max())
# print(params.shape)
# np.savez("data/data_vert.npz", simulations=simulations, parameters=params_, points=points)



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
#         self.min = torch.tensor([torch.min(params[..., i]) for i in range(params.shape[-1])])
#         self.max = torch.tensor([torch.max(params[..., i]) for i in range(params.shape[-1])])
    
#     def normalize(self, params):
#         for i in range(params.shape[-1]):
#             params[:,:,i] = (params[:,:,i] - self.min[i]) / (self.max[i] - self.min[i] + 1e-8)
#         return params

# data = np.load("data/data_vert_no.npz")
# simulations = torch.tensor(data["simulations"], dtype=torch.float32).unsqueeze(-1)
# params = torch.tensor(data["parameters"], dtype=torch.float32)
# print(params.shape, simulations.shape)

# train_size = int(0.9 * simulations.shape[0])

# u_train, u_test = simulations[:train_size], simulations[train_size:]
# p_train, p_test = params[:train_size], params[train_size:]

# lifting_net = torch.nn.Sequential(
#     torch.nn.Linear(4, 16),
#     torch.nn.Tanh(),
#     torch.nn.Linear(16, 32),
#     # torch.nn.Tanh(),
#     # torch.nn.Linear(64, 16),
# )

# projection_net = torch.nn.Sequential(
#     torch.nn.Linear(32, 16),
#     torch.nn.Tanh(),
#     torch.nn.Linear(16, 1),
#     # torch.nn.Tanh(),
#     # torch.nn.Linear(128, 1),
# )

# model = FNO(
#     lifting_net=lifting_net,
#     projecting_net=projection_net,
#     n_modes=8,
#     dimensions=2,
#     padding=8,
#     padding_type="constant",
#     inner_size=32,
#     n_layers=2,
#     func=torch.nn.Tanh,
# )

# u_normalizer = Normalizer(u_train)
# u_train, u_test = u_normalizer.normalize(u_train), u_normalizer.normalize(u_test)
# p_normalizer = NormalizerParameters(p_train)
# p_train, p_test = p_normalizer.normalize(p_train), p_normalizer.normalize(p_test)

# print(u_train.max(), u_train.min())
# print(p_train.max(), p_train.min())

# problem = SupervisedProblem(
#     input_= p_train,
#     output_= u_train
# )

# optimizer = TorchOptimizer(torch.optim.AdamW, lr=1e-3)

# solver = SupervisedSolver(
#     model = model,
#     problem = problem,
#     optimizer = optimizer,
#     use_lt=False
# )

# es = EarlyStopping(
#     monitor='val_loss',
#     patience=100,
#     mode='min',
#     verbose=True
# )

# checkpoint = ModelCheckpoint(
#     monitor='val_loss',
#     mode='min',
#     save_top_k=1,
#     filename='best_model',
#     save_weights_only=True
# )
# logger = TensorBoardLogger(
#     save_dir='logs',
#     name="fno",
# )
# trainer = Trainer( 
#     solver=solver,
#     max_epochs= 100000,
#     batch_size= 128,
#     train_size= 0.9,
#     val_size= 0.1,
#     accelerator= 'cuda',
#     devices= 1,
#     log_every_n_steps= 0,
#     callbacks=[es, checkpoint],
#     logger=logger,
# )

# trainer.train()
