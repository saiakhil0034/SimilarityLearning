import numpy as np
import torch
from itertools import combinations

from utils.distance_utils import pdist


class TripletSelector(object):

    def __init__(self, margin, negative_selection_fn="semihard", cpu=True):
        super(TripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        if negative_selection_fn == "semihard":
            self.negative_selection_fn = self.semihard_negative
        elif negative_selection_fn == "hard":
            self.negative_selection_fn = self.hardest_negative
        else:
            self.negative_selection_fn = self.random_hard_negative

    def hardest_negative(self, loss_values):
        hard_negative = np.argmax(loss_values)
        return hard_negative if loss_values[hard_negative] > 0 else None

    def semihard_negative(self, loss_values):
        semihard_negatives = np.where(np.logical_and(
            loss_values < self.margin, loss_values > 0))[0]
        return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None

    def random_hard_negative(self, loss_values):
        hard_negatives = np.where(loss_values > 0)[0]
        return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            # All anchor-positive pairs
            anchor_positives = list(combinations(label_indices, 2))
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:,
                                                            0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array(
                    [anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append(
                        [anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append(
                [anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)
