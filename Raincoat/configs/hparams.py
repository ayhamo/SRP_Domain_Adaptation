
## The cuurent hyper-parameters values are not necessarily the best ones for a specific risk.
def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class WISDM():
    def __init__(self):
        super(WISDM, self).__init__()
        self.train_params = {
                'num_epochs': 450,
                'corr_epochs': 55,
                'batch_size': 32,
                'weight_decay': 1e-4,
                'learning_rate':1e-3,
                'scheduler_steps': 32,
                'coscheduler_steps': 32,
        }

class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
                'num_epochs': 450,
                'corr_epochs': 55,
                'batch_size': 32,
                'weight_decay': 1e-4,
                'learning_rate':5e-4,
                'scheduler_steps': 32,
                'coscheduler_steps': 32,
        }


class HHAR_SA():
    def __init__(self):
        super(HHAR_SA, self).__init__()
        self.train_params = {
                'num_epochs': 450,
                'corr_epochs': 55,
                'batch_size': 32,
                'weight_decay': 1e-4,
                'learning_rate':1e-3,
                'scheduler_steps': 32,
                'coscheduler_steps': 32,
        }


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
                'num_epochs': 450,
                'corr_epochs': 55,
                'batch_size': 128,
                'weight_decay': 1e-4,
                'learning_rate':0.002,
                'scheduler_steps': 32,
                'coscheduler_steps': 32,
        }

class WISDM2HHAR_SA():
    def __init__(self):
        super(WISDM2HHAR_SA, self).__init__()
        self.train_params = {
                'num_epochs': 450,
                'corr_epochs': 55,
                'batch_size': 128,
                'weight_decay': 1e-4,
                'learning_rate':0.002,
                'scheduler_steps': 32,
                'coscheduler_steps': 32,
        }