from utils.losses import TripletLoss
from utils.triplet_construction import TripletSelector
from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low=0)
margin = 0.2
config = {
    "input_fl": 128,
    "output_fl": 64,
    "lr": 1e-3,
    "gamma": 0.1,
    "num_epochs": 200,
    "log_interval": 10,
    "lrstep_interval": 50,
    "num_classes_batch": 10,
    "num_samples_class": 20,
    "num_workers": 2,
    "train_seq": [0, 2, 3, 4, 5, 7, 9, 11, 17, 20],
    "test_seq": [1, 6, 8, 10, 12, 13, 14, 15, 16, 18, 19],
    "transforms": [],
    # "loss_fn": TripletLoss(margin, TripletSelector(margin, "semihard")),
    "accuracy_calculator": AccuracyCalculator(include=("precision_at_1",), k=1),
    "loss_fn": losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer),
    "mining_fn": miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="semihard")
}
