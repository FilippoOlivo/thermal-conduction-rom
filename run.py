import os
import yaml
import importlib
from pina import Trainer
import torch
import numpy as np
from pina.problem.zoo import SupervisedProblem
from pina.solver import ReducedOrderModelSolver
from pina.solver import SupervisedSolver
from pina.optim import TorchOptimizer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from copy import deepcopy
from rom.normalization import Normalizer, NormalizerParameters


def compute_error(u_true, u_pred):
    """Compute the L2 error between true and predicted solutions."""
    return np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true)

def argparse():
    import argparse
    parser = argparse.ArgumentParser(description="Train a model with specified "
        "parameters.")
    parser.add_argument('--config', type=str, required=True, 
        help='Path to the configuration YAML file.')
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
    points = data['points']
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
    return u_train, u_test, p_train, p_test, normalizer_sims, points
    
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
        save_weights_only=True
    )
    logger = TensorBoardLogger(
        save_dir=trainer_args.pop('log_dir', 'logs'),
        name=trainer_args.pop('name'),
        version=f"{trainer_args.pop('version'):03d}"
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

def save_model(solver, trainer, problem, model, int_net):
    model_path = trainer.logger.log_dir.replace("logs", "models")
    os.makedirs(model_path, exist_ok=True)
    if int_net is None:
        solver = SupervisedSolver.load_from_checkpoint(
            os.path.join(trainer.logger.log_dir, 'checkpoints', 
                'best_model.ckpt'), 
            problem=problem, 
            model=model, 
            use_lt=False)
        model = solver.model.cpu()
        model.eval()
        torch.save(model.state_dict(), os.path.join(model_path, 'model.pth'))
        if hasattr(model, 'pod'):
            torch.save(model.pod.basis, os.path.join(model_path, 'pod_basis.pth'))
    else: 
        solver = ReducedOrderModelSolver.load_from_checkpoint(
            os.path.join(trainer.logger.log_dir, 'checkpoints', 
                'best_model.ckpt'), 
            problem=problem, 
            interpolation_network=int_net, 
            reduction_network=model
        )
        int_net = solver.model["interpolation_network"].cpu()
        torch.save(int_net.state_dict(), os.path.join(model_path, 'interpolation_network.pth'))
        model = solver.model["reduction_network"].cpu()
        torch.save(model.state_dict(), os.path.join(model_path, 'reduction_network.pth'))
        if hasattr(model, 'pod'):
            torch.save(model.pod.basis, os.path.join(model_path, 'pod_basis.pth'))


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
    u_train, u_test, p_train, p_test, normalizer_sims, points = load_data(data_args)
    problem = SupervisedProblem(output_=u_train, input_=p_train)
    optimizer = load_optimizer(config.get("optimizer", {}))
    if int_net is None:
        if hasattr(model, 'fit_pod'):
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
    save_model(solver, trainer, problem, model, int_net)

if __name__ == "__main__":
    main()

