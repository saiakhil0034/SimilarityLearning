from utils.losses import TripletLoss
from utils.triplet_construction import TripletSelector

margin = 1
config = {
    "lr" = 1e-3,
    "gamma" = 0.1
    "num_epochs" = 100,
    "log_interval" = 100,
    "lrstep_interval" = 10,
    "num_classes_batch" = 10,
    "num_samples_class" = 10,
    "num_workers" = 2,
    "loss_fn" = TripletLoss(margin=margin, TripletSelector(margin, "semihard"))
}
