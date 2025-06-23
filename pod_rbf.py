import torch 
import numpy as np
from pina.model.block import PODBlock, RBFBlock
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import h5py

def compute_error(u_true, u_pred):
    """Compute the L2 error between true and predicted solutions."""
    return np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true)


with h5py.File('solution.h5', 'r') as f:
    x, y = f['x'][:], f['y'][:]  
data = np.load("data_vert.npz")
simulations = data['simulations']
params = data['parameters']

p_train = torch.tensor(params[:400], dtype=torch.float64)
u_train = torch.tensor(simulations[:400], dtype=torch.float64)

p_test = torch.tensor(params[400:], dtype=torch.float64)
u_test = torch.tensor(simulations[400:], dtype=torch.float64)


# POD-RBF model class
class PODRBF(torch.nn.Module):
    def __init__(self, pod_rank, rbf_kernel):
        super().__init__()
        self.pod = PODBlock(pod_rank, scale_coefficients=False)
        self.rbf = RBFBlock(kernel=rbf_kernel)
        

    def forward(self, x):
        coefficients = self.rbf(x)
        return self.pod.expand(coefficients)

    def fit(self, p, x):
        self.pod.fit(x, randomized=False)
        self.rbf.fit(p, self.pod.reduce(x))
        self.rbf._coeffs = self.rbf._coeffs.to(torch.float64)
        
pod_rbf = PODRBF(pod_rank=250, rbf_kernel="thin_plate_spline").double()

# fit POD-RBF model
pod_rbf.fit(p_train, u_train)

pod_tmp = pod_rbf.pod
pod_train = pod_tmp.expand(pod_tmp.reduce(u_test))
u_pred = pod_rbf(p_test)

sv = pod_rbf.pod.singular_values / pod_rbf.pod.singular_values[0]
plt.semilogy(sv, marker='o')
plt.savefig("singular_values.png")
plt.clf()


tria = Triangulation(x, y)

fig, ax = plt.subplots(1, 3, figsize=(12, 5))

# Prepare data
podrbf = u_pred[0].detach().cpu().numpy()
utrue = u_test[0].detach().cpu().numpy()
diff = np.abs(podrbf - utrue)

levels_main = np.linspace(0, utrue.max(), 100)
levels_diff = np.linspace(0, diff.max(), 100)

# POD-RBF plot
ax[0].set_title("POD-RBF")
tcf0 = ax[0].tricontourf(tria, podrbf, cmap='jet', levels=levels_main)
fig.colorbar(tcf0, ax=ax[0])

# True solution plot
ax[1].set_title("True")
tcf1 = ax[1].tricontourf(tria, utrue, cmap='jet', levels=levels_main)
fig.colorbar(tcf1, ax=ax[1])

# Difference plot
ax[2].set_title("Difference")
tcf2 = ax[2].tricontourf(tria, diff, cmap='jet', levels=levels_diff)
fig.colorbar(tcf2, ax=ax[2])

# Optional formatting
for a in ax:
    a.set_aspect('equal')

# plt.tight_layout()
fig.text(0.5, 0.02, rf"$\mathrm{{Error}} = {compute_error(u_test, u_pred):.6f}$", ha='center', fontsize=12)
plt.savefig("pod_rbf_comparison.png", dpi=300)
plt.close()

