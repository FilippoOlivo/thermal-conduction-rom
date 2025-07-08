from build.thermal_conduction import ThermalConduction
import random
from tqdm import tqdm
import numpy as np

# Define problem
regions = [2.0, 4.0]
conductivities = [20.0, 10.0, 1.0]

problem = ThermalConduction(regions=regions, 
                            conductivities=conductivities,
                            axis=1,
                            boundary_temperatures=[0.0, 0.0, 1.0, -1.0]
                           )

# Sample temperatures
min_, max_ = [.5, 3.75], [3.25, 5.5]
regions = [[random.uniform(min_[i], max_[i]) for i in range(len(regions))] for _ in range(2000)]
min_, max_ = 100, 1000
temperatures = [random.uniform(min_, max_) for _ in range(2000)]


def simulate(regions, temperatures):
    simulations= []
    boundary_temperatures = [0,0,1,-1]
    parameters = []
    for r, t in tqdm(zip(regions, temperatures), total=len(regions)):
        problem.set_regions(r)
        boundary_temperatures[2] = t
        parameters.append([r[0], r[1], t])
        problem.set_boundary_temperatures(boundary_temperatures)
        simulations.append(problem.solve_system())
    return np.array(simulations), np.array(parameters)

s,p = simulate(regions, temperatures)
np.savez('data/data_vert.npz', simulations=s, parameters=p)


 
# Define problem
# regions = [1.0, 2.0]
# conductivities = [1.0, 1.0, 1.0]
# axis = 0
# boundary_temperatures = [0.0, 0.0, 500.0, -1.0]
# problem = ThermalConduction(regions=regions, 
#                             conductivities=conductivities,
#                             axis=0,
#                             boundary_temperatures=boundary_temperatures
#                            )


# # Sample temperatures
# min_, max_ = 0.1, 10
# conductivities = [[random.uniform(min_, max_) for _ in range(len(conductivities))] for _ in range(1500)]

# def simulate(conductivities):
#     simulations= []
#     for conductivity in tqdm(conductivities):
#         problem.set_conductivities(conductivity)
#         simulations.append(problem.solve_system())
#     return simulations
    
# s,p = simulate(conductivities=conductivities), np.array(conductivities)
# np.savez('data_hor.npz', simulations=s, parameters=p)