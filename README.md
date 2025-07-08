# Advanced reduced order models in scientific machine learning

This repo contains the code of the project valid for the exam of "Advanced reduced order models in scientific machine learning".

## How to run the code

To run the code, you need to have Python 3.8 or higher installed on your machine. You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

Moreover, to run the simulation you need to have Deal.II installed on your machine. You can find the instructions to install it [here](https://www.dealii.org/). The recommended version is 9.6.2.

## Repo structure

The repo is structured as follows:

- `src/`: contains the `deal.II` source code of the project.
- `experiments/`: contains the YAML files to run the experiments.
- `rom/`: contains the models and the code to run the reduced order models.
- `run.py`: the script to run a single the experiments.
- `submit_train.sh`: the script to submit the training of multiple experiments.
- `save_data.py`: the script to save the data from high fidelity simulations.
- `report.ipynb`: the Jupyter notebook with the report of the project.
- `requirements.txt`: the file with the required packages to run the code.
- `README.md`: this file.
- `CMakeLists.txt`: the CMake file to build code with `deal.II`.

## Report and results

Main results and the report of the project can be found in the `report.ipynb` file.
