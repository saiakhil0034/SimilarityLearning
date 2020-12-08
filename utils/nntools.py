import os
import sys
import torch
from torch import nn
import torch.utils.data as td
from utils.dataset import get_loader, BalancedBatchSampler
from config import config
from matplotlib import pyplot as plt
from time import time
from pprint import pprint
from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


class StatsManager(object):
    """
    A class meant to track the loss during a neural network learning experiment.

    Though not abstract, this class is meant to be overloaded to compute and
    track statistics relevant for a given task. For instance, you may want to
    overload its methods to keep track of the accuracy, top-5 accuracy,
    intersection over union, PSNR, etc, when training a classifier, an object
    detector, a denoiser, etc.
    """

    def __init__(self):
        self.init()

    def __repr__(self):
        """Pretty printer showing the class name of the stats manager. This is
        what is displayed when doing ``print(stats_manager)``.
        """
        return self.__class__.__name__

    def init(self):
        """Initialize/Reset all the statistics"""
        self.running_loss = 0
        self.number_update = 0

    def accumulate(self, loss, num, x=None, y=None, d=None):
        """Accumulate statistics

        Though the arguments x, y, d are not used in this implementation, they
        are meant to be used by any subclasses. For instance they can be used
        to compute and track top-5 accuracy when training a classifier.

        Arguments:
            loss (float): the loss obtained during the last update.
            x (Tensor): the input of the network during the last update.
            y (Tensor): the prediction of by the network during the last update.
            d (Tensor): the desired output for the last update.
        """
        self.running_loss += loss
        self.number_update += num

    def summarize(self):
        """Compute statistics based on accumulated ones"""
        return self.running_loss / self.number_update


class Experiment(object):
    """
    A class meant to run a neural network learning experiment.

    After being instantiated, the experiment can be run using the method
    ``run``. At each epoch, a checkpoint file will be created in the directory
    ``output_dir``. Two files will be present: ``checkpoint.pth.tar`` a binary
    file containing the state of the experiment, and ``config.txt`` an ASCII
    file describing the setting of the experiment. If ``output_dir`` does not
    exist, it will be created. Otherwise, the last checkpoint will be loaded,
    except if the setting does not match (in that case an exception will be
    raised). The loaded experiment will be continued from where it stopped when
    calling the method ``run``. The experiment can be evaluated using the method
    ``evaluate``.

    Attributes/Properties:
        epoch (integer): the number of performed epochs.
        history (list): a list of statistics for each epoch.
            If ``perform_validation_during_training``=False, each element of the
            list is a statistic returned by the stats manager on training data.
            If ``perform_validation_during_training``=True, each element of the
            list is a pair. The first element of the pair is a statistic
            returned by the stats manager evaluated on the training set. The
            second element of the pair is a statistic returned by the stats
            manager evaluated on the validation set.

    Arguments:
        net (NeuralNetork): a neural network.
        train_set (Dataset): a training data set.
        val_set (Dataset): a validation data set.
        stats_manager (StatsManager): a stats manager.
        output_dir (string, optional): path where to load/save checkpoints. If
            None, ``output_dir`` is set to "experiment_TIMESTAMP" where
            TIMESTAMP is the current time stamp as returned by ``time.time()``.
            (default: None)
        batch_size (integer, optional): the size of the mini batches.
            (default: 16)
        perform_validation_during_training (boolean, optional): if False,
            statistics at each epoch are computed on the training set only.
            If True, statistics at each epoch are computed on both the training
            set and the validation set. (default: False)
    """

    def __init__(self, model, device, cuda, args, optimizer, stats_manager,
                 output_dir=None, perform_validation_during_training=True):

        # Define data loaders
        train_loader, data_train = get_loader(cuda, args.data_path,
                                              config["train_seq"], config, shuffle=True)

        test_loader, data_test = get_loader(cuda, args.data_path,
                                            config["test_seq"], config, shuffle=False)

        self.mining_fn = config["mining_fn"]
        self.loss_fn = config["loss_fn"]
        self.accuracy_calculator = config["accuracy_calculator"]
        output_dir = f"./results/{output_dir}"

        # Initialize history
        history = {
            'losses': [],
            'val_accuracy': [],
            'best_val_loss': 100000.0,
            'best_epoch': -1
        }

        # Define checkpoint paths
        if output_dir is None:
            output_dir = f'experiment_{time.time()}'
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "checkpoint.pth.tar")
        config_path = os.path.join(output_dir, "config.txt")
        bestmodel_path = os.path.join(output_dir, "bestmodel.pth.tar")
        bestmodel_config_path = os.path.join(
            output_dir, "bestmodel_config.txt")
        plot_path = os.path.join(output_dir, "loss_plot.png")

        # Transfer all local arguments/variables into attributes
        locs = {k: v for k, v in locals().items() if k is not 'self'}
        self.__dict__.update(locs)

        # Load checkpoint and check compatibility
        if os.path.isfile(config_path):
            with open(config_path, 'r') as f:
                if f.read()[:-1] != repr(self):
                    raise ValueError(
                        "Cannot create this experiment: "
                        "I found a checkpoint conflicting with the current setting.")
            self.load()
        else:
            self.save()

    @property
    def epoch(self):
        """Returns the number of epochs already performed."""
        return len(self.history['losses'])

    def setting(self):
        """Returns the setting of the experiment."""
        return {'Model': self.model,
                'Optimizer': self.optimizer,
                'StatsManager': self.stats_manager,
                'PerformValidationDuringTraining': self.perform_validation_during_training}

    def __repr__(self):
        """Pretty printer showing the setting of the experiment. This is what
        is displayed when doing ``print(experiment)``. This is also what is
        saved in the ``config.txt`` file.
        """
        string = ''
        for key, val in self.setting().items():
            string += f'{key}({val})\n'
        return string

    def state_dict(self):
        """Returns the current state of the experiment."""
        return {'Model': self.model.state_dict(),
                'Optimizer': self.optimizer.state_dict(),
                'History': self.history}

    def load_state_dict(self, checkpoint):
        """Loads the experiment from the input checkpoint."""
        self.model.load_state_dict(checkpoint['Model'])
        self.optimizer.load_state_dict(checkpoint['Optimizer'])
        self.history = checkpoint['History']

        # The following loops are used to fix a bug that was
        # discussed here: https://github.com/pytorch/pytorch/issues/2830
        # (it is supposed to be fixed in recent PyTorch version)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    def save(self):
        """Saves the experiment on disk, i.e, create/update the last checkpoint."""
        torch.save(self.state_dict(), self.checkpoint_path)
        with open(self.config_path, 'w') as f:
            print(self, file=f)

    def save_bestmodel(self):
        """Saves the best experiment on disk"""
        torch.save(self.state_dict(), self.bestmodel_path)
        with open(self.bestmodel_config_path, 'w') as f:
            print(self, file=f)

    def load(self):
        """Loads the experiment from the last checkpoint saved on disk."""
        checkpoint = torch.load(self.checkpoint_path,
                                map_location=self.device)
        self.load_state_dict(checkpoint)
        del checkpoint

    def load_bestmodel(self):
        bestmodel = torch.load(self.bestmodel_path,
                               map_location=self.device)
        self.load_state_dict(bestmodel)
        del bestmodel

    def plot(self):
        # plots the
        trainLosses, valLosses = zip(*self.history['losses'])
        base = [i + 1 for i in list(range(len(trainLosses)))]
        plt.figure()
        plt.plot(base, trainLosses)
        plt.plot(base, valLosses)
        plt.gca().legend(('train', 'validation'))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs Epochs')
        plt.savefig(self.plot_path)

    def run(self, num_epochs, plot=None):
        """Runs the experiment, i.e., trains the network using backpropagation
        based on the optimizer and the training set. Also performs statistics at
        each epoch using the stats manager.

        Arguments:
            num_epoch (integer): the number of epoch to perform.
            plot (func, optional): if not None, should be a function taking a
                single argument being an experiment (meant to be ``self``).
                Similar to a visitor pattern, this function is meant to inspect
                the current state of the experiment and display/plot/save
                statistics. For example, if the experiment is run from a
                Jupyter notebook, ``plot`` can be used to display the evolution
                of the loss with ``matplotlib``. If the experiment is run on a
                server without display, ``plot`` can be used to show statistics
                on ``stdout`` or save statistics in a log file. (default: None)
        """
        self.model.train()
        self.stats_manager.init()
        start_epoch = self.epoch
        device = self.device
        min_val_loss = self.history['best_val_loss']

        print("Start/Continue training from epoch {}".format(start_epoch))
        if plot is not None:
            plot(self)
        for epoch in range(start_epoch, num_epochs):
            s = time()
            # self.stats_manager.init()
            # for batch_idx, (data, target) in enumerate(train_loader):
            self.train_epoch()

            print(f'Time taken for train : {time() - s}')
            if not self.perform_validation_during_training:
                self.history.append(self.stats_manager.summarize())
            else:
                # train_loss = self.stats_manager.summarize() #don't change the order
                train_loss = self.stats_manager.summarize()
                start = time()
                val_loss, accuracy = self.evaluate(
                    mode='val', generate=False)  # don't change the order
                end = time()
                message += f'\nTime taken for validation : {end - start}'
                message += f'\nVal accuracy (Precision@1) at {epoch} : {accuracy}'
                message += f'\nVal Loss at {epoch} : {val_loss}'
                message += f'\nTrain Loss at {epoch} : {train_loss}'
                print(message)
                self.history['losses'].append((train_loss, val_loss))
                self.history['val_accuracy'].append(accuracy)
                if(val_loss < min_val_loss):
                    min_val_loss = val_loss
                    self.save_bestmodel()
                    self.history['best_val'] = min_val_loss
                    self.history['best_epoch'] = epoch
                    print('Best model saved with Val loss', min_val_loss)
                    self.history['test_accuracy'].append(accuracy)
                with open(os.path.join(self.output_dir, 'history.json'), 'w') as f:
                    json.dump(self.history, f)

            self.save()
            self.plot()
            # if plot is not None:
            #     plot(self)
        print("Finish training for {} epochs".format(num_epochs))

    def batch_forward_pass(self, data, labels):
        data, labels = data.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        embeddings = self.model(data)
        indices_tuple = self.mining_fn(embeddings, labels)
        loss = self.loss_fn(embeddings, labels, indices_tuple)

        return loss, self.mining_fn.num_triplets, data[0].shape[0]

    # def batch_forward_pass(self, data, target):
    #     target = target if len(target) > 0 else None
    #     # if not type(data) in (tuple, list):
    #     #     data = (data,)
    #     data = data.to(self.device)

    #     # data = tuple(d.to(self.device) for d in data)
    #     if target is not None:
    #         target = target.to(self.device)

    #     embeddings = self.model(data)

    #     # if type(outputs) not in (tuple, list):
    #     #     outputs = (outputs,)
    #     # loss_inputs = outputs
    #     # if target is not None:
    #     #     target = (target,)
    #     #     loss_inputs += target

    #     indices_tuple = self.mining_fn(embeddings, target)
    #     loss = self.loss_fn(embeddings, target, indices_tuple)
    #     # loss, num_triplets = self.loss_fn(*loss_inputs)
    #     return loss, mining_func.num_triplets, data[0].shape[0]

    def train_epoch(self):
        self.stats_manager.init()
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            loss, num_triplets, num_elems = self.batch_forward_pass(
                data, target)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.stats_manager.accumulate(loss.item(), num_triplets)

        return self.stats_manager.summarize(), loss.item()

    def get_all_embeddings(self, dataset):
        tester = testers.BaseTester()
        return tester.get_all_embeddings(dataset, self.model)

    def evaluate(self):
        """Evaluates the experiment, i.e., forward propagates the validation set
        through the network and returns the statistics computed by the stats
        manager.
        """
        self.stats_manager.init()
        self.model.eval()
        device = self.device
        loaderToRun = self.test_loader
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loaderToRun):
                loss, num_triplets, num_elems = self.batch_forward_pass(
                    data, target)
                self.stats_manager.accumulate(loss.item(), num_triplets)

            train_embeddings, train_labels = self.get_all_embeddings(
                self.data_train)
            test_embeddings, test_labels = self.get_all_embeddings(
                self.data_test)
            print("Computing accuracy")
            accuracies = self.accuracy_calculator.get_accuracy(test_embeddings,
                                                               train_embeddings,
                                                               np.squeeze(
                                                                   test_labels),
                                                               np.squeeze(
                                                                   train_labels),
                                                               False)
            print(
                "Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))

        return self.stats_manager.summarize(), accuracies["precision_at_1"]
