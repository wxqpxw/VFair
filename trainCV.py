import copy
import os
import random

import torch.autograd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adagrad, Adam, lr_scheduler
from cv_dataloader import *
from dataloader import TensorBoardLogger
from cv_DRO import DRO
from cv_ARL import ARL
from cv_VFair import Var as VFair
from resnet import baseline as ERM
from metrics import FairnessMetrics
from argparser import get_args, DefaultArguments
from tqdm import tqdm
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

def train_model(
        model,
        train_loader,
        test_loader,
        train_epoch,
        pretrain_steps,
        optimizer_learner,
        optimizer_adv,
        metrics,
        checkpoint_dir,
        logger_learner,
        logger_metrics,
        n_iters,
        device="cpu",
        regression=True
):
    """
    Function for training the model on a dataset for a single epoch.

    Args:
        model: model to train.
        train_loader: Data Loader for the dataset you want to train on.
        test_dataset: Data iterator of the test set.
        train_epoch: Number of training epochs.
        pretrain_steps: Number of pretrain steps (no adversary training).
        optimizer_learner: The optimizer used to update the learner.
        optimizer_adv: The optimizer used to update the adversary.
        metrics: Metrics objects for saving AUC (see metrics.py).
        checkpoint_dir: Directory to save the tensorboard checkpoints.
        logger_learner: Object in which loss learner curves are stored.
        logger_adv: Object in which loss learner curves are stored (adversary).
        logger_metrics: Objects in which AUC metrics are stored.
        n_iters: How often to train the model with different seeds.
        device: device to train on (cpu / gpu),
    """
    model.train()
    total_steps = 0
    schedular = lr_scheduler.StepLR(optimizer_learner, train_epoch // 4, gamma=0.5)

    # Reset the dataloader if out of data.
    for _ in tqdm(range(train_epoch)):
        for step, (train_x, train_y) in enumerate(train_loader):

            # model.draw_loss(step)
            # Transfer data to GPU if possible.
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            total_steps += 1

            # Learner update step.
            loss_learner, train_logits, logging_dict = model.learner_step(
                train_x, train_y, regression=regression
            )
            logger_learner.add_values(logging_dict)
            optimizer_learner.step()

            # Adversary update step (if ARL model).
            if optimizer_adv:
                if total_steps >= pretrain_steps:
                    model.adversary_step(
                        train_x, train_logits, train_y
                    )
                    optimizer_adv.step()
                else:
                    loss_adv = -loss_learner
        schedular.step()
    # test
    target = []
    pred = []
    logi = []
    # soft = []
    for step, (test_x, test_y) in enumerate(test_loader):
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        with torch.no_grad():
            test_pred, test_logi, test_soft = model.learner_test(test_x)
        target.append(test_y)
        pred.append(test_pred)
        logi.append(test_logi)
        # soft.append(test_soft)
        torch.cuda.synchronize()
    target = torch.cat(target, dim=0)
    pred = torch.cat(pred, dim=0)
    logi = torch.cat(logi, dim=0)
    # soft = torch.cat(soft, dim=0)
    # Calculate AUC and accuracy metrics.
    # metrics.set_auc(soft, target, n_iters)
    # metrics.set_auc_other(soft, target, n_iters)
    if regression:
        metrics.set_utility(logi, target, n_iters)
        metrics.set_utility_other(logi, target, n_iters)
    else:
        metrics.set_utility(pred, target, n_iters)
        metrics.set_utility_other(pred, target, n_iters)
    metrics.set_var(logi, target, n_iters)

    # full_loss = F.cross_entropy(logi, target.to('cuda'), reduction="none")
    # correct_indices = pred == target.to('cuda')
    # right_loss = torch.mean(full_loss[correct_indices])
    # logger_metrics.add_values({"right_loss": right_loss})

    # Add the metrics to tensorboard.
    logger_metrics.add_values(metrics.logging_dict)
    torch.save(
        model.state_dict(),
        os.path.join(checkpoint_dir, "model_checkpoint.pt"),
    )


def train_for_n_iters(
        train_dataset,
        test_dataset,
        train_epoch,
        model_params,
        lr_params,
        optimizer_name,
        n_iters=1,
        pretrain_steps=250,
        seed=42,
        phi=-1000,
        log_dir="logs/",
        model_name="ARL",
        regression=False
):
    """
    Trains the model for n iterations, and averages the results.

    Args:
        train_dataset: Data iterator of the train set.
        test_dataset: Data iterator of the test set.
        train_epoch: Number of training epochs
        model_params: A dictionary with model hyperparameters.
        lr_params: A dictionary with hyperparmaeters for optimizers.
        n_iters: How often to train the model with different seeds.
        optimizer_name: Optimizer
        pretrain_steps: For ARL, number of pretrain steps (steps with no adversary).
        seed: Seed for random setting
        log_dir: Directory where to save the tensorboard loggers.
        model_name: Name of the method
    """

    # Set the device on which to train.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_params["device"] = device

    # Initiate metrics object.
    metrics = FairnessMetrics(n_iters, test_dataset.subgroup_indexes, is_regression=regression)

    # Preparation of logging directories.
    experiment_dir = log_dir + str(train_epoch)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialte TensorBoard loggers.
    summary_writer = SummaryWriter(experiment_dir)
    logger_learner = TensorBoardLogger(summary_writer, name="learner")
    logger_metrics = TensorBoardLogger(summary_writer, name="metrics")

    # repeat the experiment for n times
    for i in range(n_iters):
        print(f"Training model {i + 1}/{n_iters}")
        seed_everything(seed + i)

        # Load the train dataset as a pytorch dataloader.
        train_loader = DataLoader(
            train_dataset,
            batch_size=model_params["batch_size"],
            shuffle=True,
            drop_last=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=model_params["batch_size"],
            shuffle=True,
            drop_last=False
        )

        # Create the model.
        if model_name == "ARL":
            model = ARL(batch_size=model_params["batch_size"], num_classes=1 if regression else 4)
        elif model_name == "ERM":
            model = ERM(num_classes=1 if regression else 4)
        elif model_name == "VFair":
            from cv_VFair import Var
            model = Var(num_classes=1 if regression else 4)
        elif model_name == "DRO":
            model = DRO(phi=phi, num_classes=1 if regression else 4)
        # elif model_name == "ablation":
        #     from ablationCV import Var
        #     model = Var(phi=phi, worst_group=train_dataset.subgroup_minority)
        else:
            print("Unknown model")

        # Transfer model to correct device.
        model = model.to(device)

        optimizer_learner = torch.optim.Adagrad(
            model.parameters(), lr=lr_params["learner"]
        )
        if model_name == 'ARL':
            optimizer_adv = torch.optim.Adagrad(
                model.adversary.parameters(), lr=lr_params["adversary"]
            )
        else:
            optimizer_adv = None

        if model_name == 'ARL':
            if optimizer_name == "Adagrad":
                optimizer_adv = Adagrad(
                    model.adversary.parameters(), lr=lr_params["adversary"]
                )
            else:
                optimizer_adv = Adam(
                    model.adversary.parameters(), lr=lr_params["adversary"]
                )
        else:
            optimizer_adv = None

        # Train the model with current seeds.
        train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            train_epoch=train_epoch,
            pretrain_steps=pretrain_steps,
            optimizer_learner=optimizer_learner,
            optimizer_adv=optimizer_adv,
            metrics=metrics,
            checkpoint_dir=checkpoint_dir,
            logger_learner=logger_learner,
            logger_metrics=logger_metrics,
            n_iters=i,
            device=device,
            regression=regression
        )

    # Average results and return metrics
    metrics.average_results()
    return metrics


def seed_everything(seed):
    """
    Changes the seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    """
    Main Function for the full training loop.

    Inputs:
        args: Namespace object from the argument parser.
    """
    # Load the train and test sets
    train_dataset = loadCVDataset(
        dataset=args.dataset,
        train_or_test="train",
    )
    test_dataset = loadCVDataset(dataset=args.dataset, train_or_test="test")

    # Set the model parameters.
    model_params = {}
    model_params["batch_size"] = args.batch_size
    if args.model_name == "ARL":
        model_params["adversary_hidden_units"] = [32]

    # Set the parameters of the optimizers.
    lr_params = {}
    lr_params["learner"] = args.lr_learner
    lr_params["adversary"] = args.lr_adversary

    # Calculate the average results when training over N iterations.
    metrics = train_for_n_iters(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_epoch=args.train_epoch,
        model_params=model_params,
        lr_params=lr_params,
        optimizer_name=args.optimizer,
        n_iters=args.average_over,
        pretrain_steps=args.pretrain_steps,
        seed=args.seed,
        log_dir=args.log_dir,
        model_name=args.model_name,
        regression=args.regression
    )

    # Save the metrics to output file.
    os.makedirs(args.log_dir, exist_ok=True)
    metrics.save_metrics(args.log_dir)

    print("Done training\n")
    print("-" * 35)
    print("Results\n")
    print("      Utility  WU    MAD    TAD    VAR")
    print("Value {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(metrics.utility_avg[-1], metrics.worst_avg[-1], metrics.diff_avg[-1],
                                                      metrics.sum_avg[-1], metrics.var_avg[-1]))
    print("STDEV {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(metrics.utility_var[-1], metrics.worst_var[-1], metrics.diff_var[-1],
                                                      metrics.sum_var[-1], metrics.var_var[-1]))
    print("-" * 35 + "\n")


if __name__ == "__main__":
    default_args = DefaultArguments()
    default_args.average_over = 10
    args = copy.copy(default_args)
    args.dataset = 'ageDB'
    args.batch_size = 1024
    args.lr_learner = 0.01
    # for ARL
    args.lr_adversary = 0.01
    args.pretrain_steps = 250

    args.phi = 0.4
    args.epsilon = 1
    regression = True
    args.train_epoch = 20
    args.seed = 42
    args.regression = regression
    for model_name in ['ERM', 'DRO', 'ARL', 'VFair']:
        args.model_name = model_name
        args.log_dir = f'checkpoints/nips_score_{regression}_{args.dataset}_{args.model_name}_{args.batch_size}_{args.lr_learner}/'
        for k, v in sorted(vars(args).items()):
            print(k, '=', v)
        # Run the model.
        main(args)
