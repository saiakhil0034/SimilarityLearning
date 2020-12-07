import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin, triplet_selector):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target, size_average=True):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        p_dist = (embeddings[triplets[:, 0]] -
                  embeddings[triplets[:, 1]]).pow(2).sum(1)
        n_dist = (embeddings[triplets[:, 0]] -
                  embeddings[triplets[:, 2]]).pow(2).sum(1)
        losses = F.relu(p_dist - n_dist + self.margin)

        return losses.mean(), len(triplets)


class AverageNonzeroTripletsMetric(object):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'
