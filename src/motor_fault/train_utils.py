import copy
import numpy as np

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0):
        self.patience = patience; self.min_delta = min_delta
        self.best = np.inf; self.count = 0; self.best_state = None
    def step(self, val_loss, model):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss; self.count = 0
            self.best_state = copy.deepcopy(model.state_dict()); return False
        else:
            self.count += 1; return self.count >= self.patience

early = EarlyStopping(patience=20, min_delta=0.0)
