"""Explicit RNG seeds only; all dataset-specific recipe methods inherited unchanged."""
import random
import numpy as np
import torch

class PaperReplicateSeedMixin:
    experiment_seed = None
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        random.seed(self.experiment_seed)
        np.random.seed(self.experiment_seed)
        torch.manual_seed(self.experiment_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.experiment_seed)
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        assert self.num_epochs == 100, self.num_epochs
    def get_dataloaders(self):
        train, val = super().get_dataloaders()
        for loader, offset in ((train, 0), (val, 10000)):
            if hasattr(loader, 'seeds'):
                loader.seeds = [self.experiment_seed + offset + i for i in range(loader.num_processes)]
        return train, val
    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(f'Paper replicate seed={self.experiment_seed}; E100; inherited recipe unchanged')
