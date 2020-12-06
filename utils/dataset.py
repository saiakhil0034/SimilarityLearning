import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import os
import numpy as np

from utils.build_data import get_data


def add_jitter(data, sigma=0.01, clip=0.01):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    print("hi")
    B, N, C = data.shape
    jittered_data = torch.clip(
        sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += data
    return data


class FeatureDataset(Dataset):
    def __init__(self, data_path, seqs, transform=None, add_jitter=True):

        self.seqs = seqs
        self.data = get_data(data_path, seqs)
        self.labels = self.data["label"]
        self.transform = transform
        self.add_jitter = add_jitter

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        print("hey")
        print(idx)
        print(self.data.iloc[idx])
        f_path = self.data.iloc[idx, 0]
        print(f_path)
        feature = np.load(f_path)
        print("hello")
        label = self.data.iloc[idx, 1]

        if self.transform:
            feature = transform(feature)
        if self.add_jitter:
            feature = add_jitter(feature)
        return feature, label


def get_loader(cuda, data_path, seqs, transforms, n_classes, n_samples, num_workers, shuffle):

    dataset = FeatureDataset(data_path, seqs,  transforms)

    # We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
    batch_sampler = BalancedBatchSampler(
        dataset.labels, n_classes=n_classes, n_samples=n_samples)

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if cuda else {}
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler, **kwargs)

    return data_loader


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        print(min(len(i) for i in self.label_to_indices.values()))
        print(max(len(i) for i in self.label_to_indices.values()))

        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(
                self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                class_sample_indices = self.label_to_indices[class_][
                    self.used_label_indices_count[class_]:self.used_label_indices_count[
                        class_] + self.n_samples]
                indices.extend(class_sample_indices)
                print(class_sample_indices, len(
                    self.label_to_indices[class_]), len(self.labels))
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


class TripletData(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.train = dataset.train
        self.test = dataset.test

        if self.train:
            self.labels = self.dataset.train_labels
            self.data = self.dataset.train_data
            self.labels_set = set(self.labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.dataset.test_labels
            self.test_data = self.dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(
                             self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                             np.random.choice(
                                 list(
                                     self.labels_set - set([self.test_labels[i].item()]))
                             )
                         ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item(
            )
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(
                    self.label_to_indices[label1])
            negative_label = np.random.choice(
                list(self.labels_set - set([label1])))
            negative_index = np.random.choice(
                self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        return (img1, img2, img3), []

    def __len__(self):
        return len(self.dataset)
