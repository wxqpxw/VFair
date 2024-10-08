import argparse
import json


class DefaultArguments:
    def __init__(self):
        """
        Contains the default arguments passed to the main training function.

        Description of parameters:
            average_over: Number of iterations to average results over.
            dataset: Dataset to use: compas, uci_adult, or law_school.
            train_epoch: Number of training epochs
            pretrain_steps: Number of steps to pretrain the learner.
            batch_size: Batch size to use for training.
            optimizer: Optimizer to use.
            embedding_size: The embedding size.
            lr_learner: Learning rate for the learner
            lr_adversary: Learning rate for the ARL adversary.
            phi: Parameter for DRO.
            epsilon: Parameter for VFair.
            seed: The seed to use for reproducing the results.
            log_dir: Directory where the logs should be created.
            res_dir: Directory where the results should be created.
        """
        self.average_over = 10
        self.model_name = "VFair"
        self.dataset = "compas"
        self.train_epoch = 20
        self.pretrain_steps = 250
        self.batch_size = 32
        self.optimizer = "Adagrad"
        self.embedding_size = 32
        self.lr_learner = 0.01
        self.lr_adversary = 0.01
        self.phi = 0.1
        self.epsilon = 1
        self.seed = 42
        self.regression = True

    def update(self, new_args):
        """
        Change the class attributes given new arguments.

        Args:
            new_args: dict with {'attribute': value, [...]}.
        """
        for attr, value in new_args.items():
            setattr(self, attr, value)


def get_args():
    parser = argparse.ArgumentParser()
    default = DefaultArguments()

    parser.add_argument(
        "--average_over",
        default=10,
        type=int,
        help="Number of iterations to average results over",
    )

    # Model parameters
    parser.add_argument(
        '-m',
        "--model_name",
        default=default.model_name,
        choices=['ERM', 'DRO', 'ARL', 'VFair', 'MPFR', 'FKL'],
        type=str,
        help="Name of the model: VFair or baseline",
    )
    parser.add_argument(
        '-d',
        "--dataset",
        default=default.dataset,
        choices=['uci_adult', 'law_school', 'compas', 'crime', 'synthetic'],
        type=str,
        help="Dataset to use: uci_adult, compas, or law_school",
    )
    parser.add_argument(
        '-e',
        "--train_epoch",
        default=default.train_epoch,
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--pretrain_steps",
        default=default.pretrain_steps,
        type=int,
        help="Number of steps to pretrain the ARL learner",
    )
    parser.add_argument(
        '-b',
        "--batch_size",
        default=default.batch_size,
        type=int,
        help="Batch size to use for training",
    )
    parser.add_argument(
        "--optimizer",
        default=default.optimizer,
        type=str,
        help="Optimizer to use"
    )
    parser.add_argument(
        "--embedding_size",
        default=default.embedding_size,
        type=int,
        help="Embedding size"
    )
    parser.add_argument(
        '-l',
        "--lr_learner",
        default=default.lr_learner,
        type=float,
        help="Learning rate for the learner"
    )
    parser.add_argument(
        "--lr_adversary",
        default=default.lr_adversary,
        type=float,
        help="Learning rate for the adversary",
    )
    parser.add_argument(
        "-r",
        "--regression",
        default=default.regression,
        type=bool,
        help="doing regression or classification",
    )

    # Other hyperparameters
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed to use for reproducing results",
    )
    parser.add_argument(
        "--log_dir",
        default="logs/",
        type=str,
        help="Directory where the logs should be created",
    )
    parser.add_argument(
        "--phi",
        default=0.9,
        type=float,
        help="Parameter in DRO",
    )
    parser.add_argument(
        "--epsilon",
        default=1.0,
        type=float,
        help="Parameter in VFair"
    )
    args = parser.parse_args()
    args.log_dir = f'checkpoints/{args.regression}_{args.dataset}_{args.model_name}_{args.batch_size}_{args.lr_learner}/'

    return args
