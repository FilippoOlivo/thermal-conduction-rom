import os
import yaml
import importlib
from model.dl_rom import DLROM
from pina import Trainer
import torch
import numpy as np
from pina.problem.zoo import SupervisedProblem
from pina.solver import ReducedOrderModelSolver
from pina.solver import SupervisedSolver
from pina.optim import TorchOptimizer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import h5py
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
import json
from copy import deepcopy

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
        self.min = params.min(0, keepdim=True).values
        self.max = params.max(0, keepdim=True).values
    
    def normalize(self, params):
        return (params - self.min) / (self.max - self.min + 1e-8)


def compute_error(u_true, u_pred):
    """Compute the L2 error between true and predicted solutions."""
    return np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true)

def argparse():
    import argparse
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    return parser.parse_args()

def load_config(config_file):
    """
    Configure the training parameters.
    :param str config_file: Path to the configuration file.
    :return: Configuration dictionary.
    :rtype: dict
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_model(model_args):
    model_class = model_args.pop("model_class", "")
    module_path, class_name = model_class.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    model = cls(**model_args)
    return model

def load_data(data_args):
    data_path = data_args.get('data_path', "")
    data = np.load(data_path)
    simulations = torch.tensor(data['simulations'], dtype=torch.float32)
    params = torch.tensor(data['parameters'], dtype=torch.float32)
    normalize = data_args.get('normalize', True)
    dataset_len = data_args['dataset_length']
    simulations = simulations[:dataset_len]
    params = params[:dataset_len]
    train_size = int(dataset_len * 0.9)
    u_train, u_test = simulations[:train_size], simulations[train_size:]
    p_train, p_test = params[:train_size], params[train_size:]
    if normalize:
        normalizer_sims = Normalizer(u_train)
        normalizer_params = NormalizerParameters(p_train)
        u_train = normalizer_sims.normalize(u_train)
        p_train, p_test = normalizer_params.normalize(p_train), normalizer_params.normalize(p_test)
    else:
        normalizer_sims = None
        normalizer_params = None
    return u_train, u_test, p_train, p_test, normalizer_sims, normalizer_params
    
def load_trainer(trainer_args, solver):
    patience = trainer_args.pop("patience", 100)
    es = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min',
        verbose=True
    )
    
    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        filename='best_model',
    )
    logger = TensorBoardLogger(
        save_dir=trainer_args.pop('log_dir', 'logs'),
        name=trainer_args.pop('name'),
    )
    trainer_args['callbacks'] = [es, checkpoint]
    trainer_args['solver'] = solver
    trainer_args['logger'] = logger
    trainer = Trainer(**trainer_args)
    return trainer

def load_optimizer(optim_args):
    print("Loading optimizer with args:", optim_args)
    optim_class = optim_args.pop("optimizer_class", "")
    module_path, class_name = optim_class.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return TorchOptimizer(
        cls,
        **optim_args
    )
    
def train(trainer):
    trainer.train()
    
def test(trainer, problem, model, int_net, optimizer, u_test, p_test, norm):
    if int_net is None:
        solver = SupervisedSolver.load_from_checkpoint(
            os.path.join(trainer.logger.log_dir, 'checkpoints', 
                'best_model.ckpt'), 
            problem=problem, 
            model=model, 
            optimizer=optimizer,
            use_lt=False)
        model = solver.cpu()
        model.eval()
        u_pred = norm.unnormalize(model(p_test).detach()).cpu().numpy()
    else: 
        solver = ReducedOrderModelSolver.load_from_checkpoint(
            os.path.join(trainer.logger.log_dir, 'checkpoints', 
                'best_model.ckpt'), 
            problem=problem, 
            interpolation_network=int_net, 
            reduction_network=model, 
            optimizer=optimizer)
        int_net = solver.model["interpolation_network"].cpu()
        model = solver.model["reduction_network"].cpu()
        model.eval()
        int_net.eval()
        u_pred = int_net(p_test)
        u_pred = norm.unnormalize(model.decode(u_pred).detach()).cpu().numpy()
    print("L2 error:", compute_error(u_test.numpy(), u_pred))
    return u_pred

def plot_results(u_test, u_pred, path):
    with h5py.File('solution.h5', 'r') as f:
        x, y = f['x'][:], f['y'][:]  
    tria = Triangulation(x, y)
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Prepare data
    pred = u_pred[0]
    true = u_test[0].detach().cpu().numpy()
    diff = np.abs(pred - true)
    
    levels_true = np.linspace(true.min(), true.max(), 100)
    levels_pred = np.linspace(pred.min(), pred.max(), 100)
    levels_diff = np.linspace(0, diff.max(), 100)
    
    # POD-RBF plot
    ax[0].set_title("Prediction")
    tcf0 = ax[0].tricontourf(tria, pred, cmap='jet', levels=levels_pred)
    fig.colorbar(tcf0, ax=ax[0])
    
    # True solution plot
    ax[1].set_title("True")
    tcf1 = ax[1].tricontourf(tria, true, cmap='jet', levels=levels_true)
    fig.colorbar(tcf1, ax=ax[1])
    
    # Difference plot
    ax[2].set_title("Difference")
    tcf2 = ax[2].tricontourf(tria, diff, cmap='jet', levels=levels_diff)
    fig.colorbar(tcf2, ax=ax[2])
    
    # Optional formatting
    for a in ax:
        a.set_aspect('equal')
    
    # plt.tight_layout()
    fig.text(0.5, 0.02, rf"$\mathrm{{Error}} = {compute_error(u_test.numpy(), u_pred):.6f}$", ha='center', fontsize=12)
    
    plt.savefig(path, dpi=300)
    plt.close()

def save_hyperparameters(config, path):
    path = os.path.join(path, 'hyperparameters.json')
        
    # Save to JSON file
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

def main():
    seed_everything(1999, workers=True)
    args = argparse()
    config = load_config(args.config)
    config_ = deepcopy(config)
    model_args = config.get("model", {})
    model = load_model(model_args)
    if "interpolation" in config:
        model_args = config["interpolation"]
        int_net = load_model(model_args)
    else:
        int_net = None
    data_args = config.get("data", {})
    u_train, u_test, p_train, p_test, normalizer_sims, normalizer_params = load_data(data_args)
    problem = SupervisedProblem(output_=u_train, input_=p_train)
    optimizer = load_optimizer(config.get("optimizer", {}))
    if int_net is None:
        model.fit_pod(u_train)
        solver = SupervisedSolver(
            problem=problem,
            model=model,
            optimizer=optimizer,
            use_lt=False
        )
    else:
        if "pod" in config_["model"]["model_class"]:
            model.fit(u_train)
        solver = ReducedOrderModelSolver(
            problem= problem,
            reduction_network=model,
            interpolation_network=int_net,
            optimizer=optimizer,
        )
    trainer_args = config.get("trainer", {})
    trainer = load_trainer(trainer_args, solver)
    train(trainer)
    save_hyperparameters(config_, trainer.logger.log_dir)
    u_pred = test(trainer, problem, model, int_net, optimizer, u_test, p_test, 
        normalizer_sims)
    plot_results(u_test, u_pred, os.path.join(trainer.logger.log_dir, 
        'results.png'))


if __name__ == "__main__":
    main()

