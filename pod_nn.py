import torch 
import numpy as np
from pina.model.block import PODBlock
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import h5py
from pina import Trainer
from pina.solver import SupervisedSolver
from pina.model import FeedForward
from pina.problem.zoo import SupervisedProblem
from pina.optim import TorchOptimizer

def compute_error(u_true, u_pred):
    """Compute the L2 error between true and predicted solutions."""
    return np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true)

with h5py.File('solution.h5', 'r') as f:
    x, y = f['x'][:], f['y'][:]  
data = np.load("data_vert.npz")
simulations = data['simulations']
params = data['parameters']


p_train = torch.tensor(params[:1280], dtype=torch.float64)
u_train = torch.tensor(simulations[:1280], dtype=torch.float64)

p_test = torch.tensor(params[1280:], dtype=torch.float64)
u_test = torch.tensor(simulations[1280:], dtype=torch.float64)

problem = SupervisedProblem(input_=p_train, output_=u_train)

class PODNN(torch.nn.Module):
    def __init__(self, pod_rank, layers, func):
        super().__init__()
        self.pod = PODBlock(pod_rank, scale_coefficients=False)
        self.nn = FeedForward(
            input_dimensions=3,
            output_dimensions=pod_rank,
            layers=layers,
            func=func,
        ).double()

    def forward(self, p):
        coefficients = self.nn(p)
        return self.pod.expand(coefficients)

    def fit_pod(self, x):
        self.pod.fit(x)

pod_nn = PODNN(pod_rank=96, layers=[24, 48], func=torch.nn.Softplus)

pod_nn_solver = SupervisedSolver(
    problem=problem,
    model=pod_nn,
    optimizer=TorchOptimizer(torch.optim.Adam, lr=0.001),
    use_lt=False,
)

max_epochs = 50000
trainer = Trainer(
    solver=pod_nn_solver,
    max_epochs=max_epochs,
    batch_size=16,
    accelerator="cpu",
)
pod_nn.fit_pod(u_train)
trainer.train()

plt.plot(pod_nn.pod.singular_values, marker='o')
plt.savefig("singular_values.png")
plt.clf()


u_pred = pod_nn(p_test)

tria = Triangulation(x, y)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Prepare data
pred = u_pred[0].detach().cpu().numpy()
true = u_test[0].detach().cpu().numpy()
diff = np.abs(pred - true)

levels_main = np.linspace(0, true.max(), 100)
levels_diff = np.linspace(0, diff.max(), 100)

# POD-RBF plot
ax[0].set_title("POD-NN")
tcf0 = ax[0].tricontourf(tria, pred, cmap='jet', levels=levels_main)
fig.colorbar(tcf0, ax=ax[0])

# True solution plot
ax[1].set_title("True")
tcf1 = ax[1].tricontourf(tria, true, cmap='jet', levels=levels_main)
fig.colorbar(tcf1, ax=ax[1])

# Difference plot
ax[2].set_title("Difference")
tcf2 = ax[2].tricontourf(tria, diff, cmap='jet', levels=levels_diff)
fig.colorbar(tcf2, ax=ax[2])

# Optional formatting
for a in ax:
    a.set_aspect('equal')

# plt.tight_layout()
fig.text(0.5, 0.02, rf"$\mathrm{{Error}} = {compute_error(u_test.numpy(), u_pred.detach().numpy()):.6f}$", ha='center', fontsize=12)

plt.savefig("pod_nn_comparison.png", dpi=300)
plt.close()