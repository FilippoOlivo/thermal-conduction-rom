
import torch
import numpy as np
from pina.model.block import PODBlock

class POD:
    def __init__(self, n_modes, problem, affine=True, fixed_stiffness=False):
        self.pod_block = PODBlock(n_modes, scale_coefficients=False)
        self.problem = problem
        self.A_q = []
        self.rhs_q = []
        self.A_r = []
        self.rhs_r = []
        self.affine = affine
        self.fixed_stiffness = fixed_stiffness

    def _initialize_affine_components(self):
        self.A_q.clear()
        num_regions = self.problem.get_num_regions()
        for i in range(num_regions):
            rhs, matrix = self.problem.get_affine_components(i)
            rows, cols, values = matrix
            indices = torch.tensor(np.array([rows, cols]), dtype=torch.long)
            shape = (indices.max().item() + 1, indices.max().item() + 1)
            values_tensor = torch.tensor(np.array(values), dtype=torch.float32)
            self.A_q.append(torch.sparse_coo_tensor(indices, values_tensor, shape))
            self.rhs_q.append(torch.tensor(rhs, dtype=torch.float32))

    def _compute_reduced_stiffness_matrix(self):
        self.A_r.clear()
        V = self.pod_block.basis  # (n, r), expected orthonormal
        for A in self.A_q:
            Ar = V @ torch.sparse.mm(A, V.T)
            self.A_r.append(Ar)
    
    def _compite_reduced_rhs(self):
        self.rhs_r.clear()
        V = self.pod_block.basis
        for rhs in self.rhs_q:
            self.rhs_r.append(V @ rhs)

    def fit(self, snapshots):
        self.pod_block.fit(snapshots, randomized=False)
        if self.affine:
            self._fit_affine()

    def _fit_affine(self):
        self._initialize_affine_components()
        self._compute_reduced_stiffness_matrix()
        self._compite_reduced_rhs()
        
    def predict(self, params):
        if self.affine:
            return self._predict_affine(params)
        else:
            return self._predict_non_affine(params)
    
    def _predict_affine(self, params):
        u_r_list = []

        for param in params:
            if len(param) != len(self.A_r):
                raise ValueError(f"Expected {len(self.A_r)} parameters, got {len(param)}")

            # Create affine combination of matrices based on parameters
            if self.fixed_stiffness:
                A_r_mu = self.A_r[0]
            else:
                A_r_mu = sum(p * A for A, p in zip(self.A_r, param))
            rhs_r_mu = sum(p * rhs for rhs, p in zip(self.rhs_r, param))

            # Solve the reduced system - use direct solve for better stability
            u_r = torch.linalg.solve(A_r_mu, rhs_r_mu)
            u_r_list.append(u_r)

        U_r = torch.stack(u_r_list)

        # Reconstruct full solution from reduced coefficients
        expanded = self.pod_block.expand(U_r)
        return expanded
        
    def _predict_non_affine(self, params):
        u_r_list = []
        temperatures = [0,0,0,-1]
        self.problem.set_conductivities([20,10,1])
        for param in params:
            region = [param[0].item(), param[1].item()]
            self.problem.set_regions(region)
            temperatures[2] = param[2].item()
            self.problem.set_boundary_temperatures(temperatures)
            self.problem.run_assemble_system()
            rows, cols, values = self.problem.get_system_matrix()
            A = torch.sparse_coo_tensor(
                torch.tensor(np.array([rows, cols]), dtype=torch.long),
                torch.tensor(np.array(values), dtype=torch.float32),
                (max(rows) + 1, max(cols) + 1)
            )
            rhs = torch.tensor(self.problem.get_rhs(), dtype=torch.float32)
            A_r = self.pod_block.basis @ torch.sparse.mm(A, self.pod_block.basis.T)
            rhs_r = self.pod_block.basis @ rhs
            # Solve the reduced system - use direct solve for better stability
            u_r = torch.linalg.solve(A_r, rhs_r)
            u_r_list.append(u_r)
        U_r = torch.stack(u_r_list)
        # Reconstruct full solution from reduced coefficients
        expanded = self.pod_block.expand(U_r)
        return expanded 
            

    @property
    def x(self):
        return self.problem.get_x()

    @property
    def y(self):
        return self.problem.get_y()
        
    @property
    def singular_values(self):
        return self.pod_block.singular_values