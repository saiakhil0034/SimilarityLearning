import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import argparse

from models.embedding_learner import EmbeddingNet

from utils.losses import TripletLoss, AverageNonzeroTripletsMetric
from utils.process import training
from utils.triplet_construction import TripletSelector
from utils.build_data import get_data
from utils.dataset import get_loader, BalancedBatchSampler
from config import config
import torch.backends.cudnn as cudnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = torch.cuda.is_available()

# Training settings
parser = argparse.ArgumentParser(description='PyTorch triplet learning')
parser.add_argument('--data_path', type=str, help='input data path')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--name', type=str, help='experiment name')


def init_weights(m):
    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias.data is not None:
            m.bias.data.fill_(0.001)


def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    exp_name = args.name

    model = EmbeddingNet(ni=config["input_fl"],
                         no=config["output_fl"]).to(device)
    model.apply(init_weights)

    best_val_loss = 0
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = lr_scheduler.StepLR(
        optimizer, config["lrstep_interval"], config["gamma"], last_epoch=-1)

    train_loader = get_loader(cuda, args.data_path, config["train_seq"], config["transforms"], config["num_classes_batch"],
                              config["num_samples_class"], config["num_workers"], shuffle=True)
    test_loader = get_loader(cuda, args.data_path, config["test_seq"], config["transform"], config["num_classes_batch"],
                             config["num_samples_class"], config["num_workers"], shuffle=False)

    training(exp_name, train_loader, test_loader, model,
             config["loss_fn"], optimizer, scheduler, config["n_epochs"], cuda, config["log_interval"],
             best_val_loss, metrics=[AverageNonzeroTripletsMetric()])


if __name__ == "__main__":
    main()
