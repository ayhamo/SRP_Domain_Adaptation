
## The cuurent hyper-parameters values are not necessarily the best ones for a specific risk.
def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
                'num_epochs': 50,
                'batch_size': 32,
                'weight_decay': 1e-4,

        }
        self.alg_hparams = {
            'learning_rate':5e-4,     
            'src_cls_loss_wt': 0.5,    
            'domain_loss_wt': 0.5
        }


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
                'num_epochs': 40,
                'batch_size': 128,
                'weight_decay': 1e-4,

        }
        self.alg_hparams = {
            'learning_rate':0.002,     
            'src_cls_loss_wt': 0.5,    
            'domain_loss_wt': 0.5
        }


class WISDM():
    def __init__(self):
        super(WISDM, self).__init__()
        self.train_params = {
                'num_epochs': 250,
                'corr_epochs': 50,
                'batch_size': 32,
                'weight_decay': 1e-4,
                'learning_rate':1e-3,
                'scheduler_steps': 32,
                'coscheduler_steps': 32
        }



class HHAR_SA():
    def __init__(self):
        super(HHAR_SA, self).__init__()
        self.train_params = {
                'num_epochs': 40,
                'batch_size': 32,
                'weight_decay': 1e-4,
        }
        self.alg_hparams = {
            'learning_rate':0.001,     
            'src_cls_loss_wt': 0.5,    
            'domain_loss_wt': 0.5,
        }


class Boiler():
    def __init__(self):
        super(Boiler, self).__init__()
        self.train_params = {
                'num_epochs': 30,
                'batch_size': 32,
                'weight_decay': 1e-4,
        }
        self.alg_hparams = {
            'learning_rate': 0.0005,   
            'src_cls_loss_wt': 0.9603,  
            'domain_loss_wt':0.9238
        }
