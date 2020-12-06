from utils.losses import TripletLoss
from utils.triplet_construction import TripletSelector

margin = 1
config = {
    "input_fl": 128,
    "output_fl": 64,
    "lr": 1e-3,
    "gamma": 0.1,
    "num_epochs": 100,
    "log_interval": 100,
    "lrstep_interval": 10,
    "num_classes_batch": 20,
    "num_samples_class": 3,
    "num_workers": 2,
    "train_seq": [1, 2, 3],
    "test_seq": [4, 5, 6],
    "transforms": [],
    "loss_fn": TripletLoss(margin, TripletSelector(margin, "semihard"))
}
