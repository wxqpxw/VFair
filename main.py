import argparse
import copy
import train as train
from argparser import DefaultArguments


def single_train(dataset, method, batch_size=64, lr=0.01, phi=0.9, training_epoch=20, lr_adversary=1., seed=42,
                 epsilon=1, regression=True):
    # Load the default arguments
    default_args = DefaultArguments()

    # Change the amount of times the results are averaged here.
    default_args.average_over = 10

    args = copy.copy(default_args)

    args.dataset = dataset  # ['uci_adult', 'compas', 'law_school', 'celebA']
    args.model_name = method  # ['baseline', 'DRO', 'ARL', 'VFair']
    args.batch_size = batch_size
    args.lr_learner = lr
    # for ARL
    args.lr_adversary = lr_adversary
    args.pretrain_steps = 250

    args.phi = phi
    args.epsilon = 1
    args.log_dir = f'checkpoints/{regression}_{dataset}_{args.model_name}_{batch_size}_{lr}/'
    args.train_epoch = training_epoch
    args.seed = seed
    args.regression = regression

    for k, v in sorted(vars(args).items()):
        print(k, '=', v)

    # Train the model.
    train.main(args)


if __name__ == '__main__':
    single_train(
        dataset='uci_adult',
        method='ERM',
        batch_size=64,
        lr=0.01,
        training_epoch=20,
        regression=True
    )
