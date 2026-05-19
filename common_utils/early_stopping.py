import logging

# Get a logger for this module
logger = logging.getLogger(__name__)


class EarlyStopping:
    """ Early stopping handler to stop training when monitored metric stops improving """

    def __init__(self, patience=5, min_delta=0.0, mode='min', verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement.
            min_delta (float): Minimum change to qualify as improvement.
            mode (str): 'min' for minimizing the metric, 'max' for maximizing.
            verbose (bool): Whether to print messages for patience updates.
        """
        assert mode in ['min', 'max'], "Mode must be 'min' or 'max'."
        self.mode = mode
        self.min_delta = min_delta if mode == 'min' else -min_delta
        self.patience = patience
        self.verbose = verbose

        # internal state
        self.counter = 0
        self.best_metric = None
        self.early_stop = False

    def __call__(self, curr_metric):
        if self.best_metric is None:
            self.best_metric = curr_metric
            return False

        # Determine improvement based on mode
        if self.mode == 'max':
            improvement = curr_metric > self.best_metric + self.min_delta
            update_best = curr_metric > self.best_metric
        else:
            improvement = curr_metric < self.best_metric - self.min_delta
            update_best = curr_metric < self.best_metric

        if update_best:
            self.best_metric = curr_metric

        if improvement:
            self.counter = 0    # Reset counter on improvement
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"Increasing early stopping patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop