import os
import random
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adagrad, Adam, lr_scheduler
from dataloader import *
from ARL import ARL
from DRO import DRO
from ERM import ERM
from VFair import VFair
from MPFR import MP_Fair_regression
from FKL import Fair_kernel_learning
from metrics import FairnessMetrics
from argparser import get_args


def train_model(
        model,
        train_loader,
        test_dataset,
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
    test_cat, test_num, test_target = test_dataset[:]
    model.train()
    total_steps = 0
    schedular = lr_scheduler.StepLR(optimizer_learner, train_epoch // 4, gamma=0.5)

    # Reset the dataloader if out of data.
    for _ in range(train_epoch):
        for _, (train_cat, train_num, train_target) in enumerate(
                train_loader
        ):
            # model.draw_loss(step)
            # Transfer data to GPU if possible.
            if train_cat is not None:
                train_cat = train_cat.to(device)
            train_num = train_num.to(device)
            train_target = train_target.to(device)
            total_steps += 1

            # Learner update step.
            loss_learner, train_logits, logging_dict = model.learner_step(
                train_cat, train_num, train_target, regression=regression
            )
            logger_learner.add_values(logging_dict)
            optimizer_learner.step()

            # Adversary update step (if ARL model).
            if optimizer_adv:
                if total_steps >= pretrain_steps:
                    model.adversary_step(
                        train_cat, train_num, train_logits, train_target, regression
                    )
                    optimizer_adv.step()
                else:
                    loss_adv = -loss_learner
        if test_cat is not None:
            test_cat = test_cat.to(device)
        test_num = test_num.to(device).to(dtype=torch.float32)
        test_target = test_target.to(device)

        with torch.no_grad():
            test_logits, test_sigmoid, test_pred = model.learner(
                test_cat, test_num
            )

        # evaluate stage
        if regression:
            metrics.set_utility(test_logits, test_target, n_iters)
            metrics.set_utility_other(test_logits.squeeze(), test_target, n_iters)
        else:
            metrics.set_utility(test_pred, test_target, n_iters)
            metrics.set_utility_other(test_pred, test_target, n_iters)
        metrics.set_var(test_logits, test_target, n_iters)

        logger_metrics.add_values(metrics.logging_dict)
        schedular.step()
    torch.save(
        model.state_dict(),
        os.path.join(checkpoint_dir, f"model_checkpoint{n_iters}.pt"),
    )


def train_for_n_iters(
        train_dataset,
        test_dataset,
        train_epoch,
        model_params,
        lr_params,
        optimizer_name,
        n_iters=10,
        pretrain_steps=250,
        seed=42,
        log_dir="logs/",
        model_name="VFair",
        regression=True
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
            drop_last=True,
            collate_fn=general_collate,
        )

        # Create the model.
        if model_name == "ARL":
            model = ARL(
                embedding_size=model_params["embedding_size"],
                n_num_cols=model_params["n_num_cols"],
                batch_size=model_params["batch_size"],
                device=device
            )
        elif model_name == "ERM":
            model = ERM(
                embedding_size=model_params["embedding_size"],
                n_num_cols=model_params["n_num_cols"],
                device=device
            )
        elif model_name == "VFair":
            model = VFair(
                embedding_size=model_params["embedding_size"],
                n_num_cols=model_params["n_num_cols"],
                device=device,
                train_dataset=train_dataset,
                epsilon=model_params["epsilon"],
            )
        elif model_name == "DRO":
            model = DRO(
                embedding_size=model_params["embedding_size"],
                n_num_cols=model_params["n_num_cols"],
                device=device,
                phi=model_params["phi"]
            )
        else:
            device = 'cpu'
            if train_dataset.categorical_data is None:
                x_train = train_dataset.numerical_data.to(device)
                x_test = test_dataset.numerical_data.to(device)
            else:
                x_train = torch.cat((train_dataset.categorical_data, train_dataset.numerical_data), dim=1).to(device)
                x_test = torch.cat((test_dataset.categorical_data, test_dataset.numerical_data), dim=1).to(device)
            s_train = train_dataset.protected_data.to(device)
            y_train = train_dataset.target_data.to(device)

            y_test = test_dataset.target_data.to(device)
            if model_name == "MPFR":
                model = MP_Fair_regression(x_train, s_train, y_train, device=device)
            elif model_name == "FKL":
                model = Fair_kernel_learning(x_train, s_train, y_train, eta=1000, device=device)
            else:
                raise AttributeError("Unknown model")
            w_ = model.fit()
            y_pred = model.pred(x_test)
            metrics.set_utility(y_pred, y_test, i)
            metrics.set_var(y_pred, y_test, i)
            metrics.set_utility_other(y_pred, y_test, i)
            break


        # Transfer model to correct device.
        model = model.to(device)

        # optimizer.
        if optimizer_name == "Adagrad":
            optimizer_learner = Adagrad(
                model.learner.parameters(), lr=lr_params["learner"]
            )
        elif optimizer_name == "Adam":
            optimizer_learner = Adam(
                model.learner.parameters(), lr=lr_params["learner"]
            )
        else:
            raise AttributeError("Unknown optimizer")

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

        train_model(
            model=model,
            train_loader=train_loader,
            test_dataset=test_dataset,
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


def general_collate(batch):
    x_num = torch.stack([item[1] for item in batch])
    x_target = torch.stack([item[2] for item in batch])
    if batch[0][0] is None:
        return None, x_num.to(dtype=torch.float32), x_target.to(dtype=torch.float32)
    else:
        return torch.stack([item[0] for item in batch]), x_num, x_target


def main(args):
    """
    Main Function for the full training loop.

    Inputs:
        args: Namespace object from the argument parser.
    """
    # Load the train and test sets
    train_dataset = loadDataset(
        dataset=args.dataset,
        train_or_test="train",
        embedding_size=args.embedding_size
    )
    test_dataset = loadDataset(dataset=args.dataset, train_or_test="test")

    # Set the model parameters.
    model_params = {"learner_hidden_units": [64, 32], "batch_size": args.batch_size,
                    "embedding_size": train_dataset.categorical_embedding_sizes,
                    "n_num_cols": train_dataset.numerical_sizes}
    if args.model_name == "ARL":
        model_params["adversary_hidden_units"] = [32]
    elif args.model_name == "DRO":
        model_params["phi"] = args.phi
    elif args.model_name == "VFair":
        model_params["epsilon"] = args.epsilon
    elif args.model_name == "MPFR" or args.model_name == "FKL":
        args.average_over = 1

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
    # Get the default and command line arguments.
    args = get_args()

    # Run the model.
    main(args)
