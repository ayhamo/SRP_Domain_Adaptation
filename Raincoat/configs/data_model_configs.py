def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class HAR():
    def __init__(self):
        super(HAR, self)
        self.scenarios = [("2", "11"), ("6", "23"),("7", "13"),("9", "18"),("12", "16"),\
            ("13", "19"),  ("18", "21"), ("20", "6"),("23", "13"),("24", "12")]
        #self.scenarios = [("24", "12"), ("6","23")]
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # CNN and RESNET features
        
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6
        self.fourier_modes = 64
        self.out_dim = self.final_out_channels+ self.fourier_modes * 2

class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        # data parameters
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.sequence_len = 3000
        self.scenarios = [("0", "11"), ("2", "5"), ("12", "5"), ("7", "18"), ("16", "1"), ("9", "14"),\
            ("4", "12"),("10", "7"),("6", "3"),("8", "10")]
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.2

        # features
        self.mid_channels = 32
        self.final_out_channels = 128
        self.features_len = 1
        self.fourier_modes = 300
        self.out_dim = 256


class WISDM(object):
    def __init__(self):
        super(WISDM, self).__init__()
        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.sequence_len = 128
        # Closed Set DA
        self.scenarios = [("2", "32"), ("4", "15"),("7", "30"),('12','17'), ('12','19'),('18','20'),\
                          ('20','30'), ("21", "31"),("25", "29"), ('26','2')]
        
        self.H_scenarios = [('3', '2'), ('3', '7'),('13', '15'), ('14', '19'),('27', '28'), ('1', '0'),
                            ('1', '3'), ('10', '11'),('22', '17'), ('27', '15')]
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.fourier_modes = 64
        self.fraction_order = 0.4
        
        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.out_dim = self.final_out_channels+ self.fourier_modes * 2

        self.features_len = 1


class HHAR_SA(object):  ## HHAR dataset, SAMSUNG device.
    def __init__(self):
        super(HHAR_SA, self).__init__()
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.sequence_len = 128

        self.scenarios = [("0", "2"), ("1", "6"),("2", "4"),("4", "0"),("4", "5"),\
            ("5", "1"),("5", "2"),("7", "2"),("7", "5"),("8", "4")]
        
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = True
        self.normalize = True
        self.fourier_modes = 32

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5

        # features
        self.mid_channels = 64 * 2
        self.final_out_channels = 128
        self.features_len = 1
        self.out_dim = self.final_out_channels+ self.fourier_modes * 2

class WISDM2HHAR_SA(object):
    def __init__(self):
        super(WISDM2HHAR_SA, self).__init__()
        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.sequence_len = 128
        
        # UniDA        
        self.H_scenarios = [('4', '0'), ('5', '1'),('6', '2'), ('7', '3'),('17', '4'), ('18', '5'),
                            ('19', '6'), ('20', '7')] # Last scenario ('23', '8') does not even work :)
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        #CNN
        self.mid_channels = 64 #*2 hhar
        self.final_out_channels = 128
        self.features_len = 1

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.fourier_modes = 64
        self.out_dim = self.final_out_channels+ self.fourier_modes * 2

        # change configs
        self.fraction_order = 0.4

class HHAR_SA2WISDM(object): 
    def __init__(self):
        super(HHAR_SA2WISDM, self).__init__()
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.sequence_len = 128

        self.scenarios = [("0", "4"), ("1", "5"),("2", "6"),("3", "7"),("4", "17"),\
            ("5", "18"),("6", "19"),("7", "20"),("8", "23")]
        
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = True
        self.normalize = True
        self.fourier_modes = 32

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5

        # features
        self.mid_channels = 64 * 2
        self.final_out_channels = 128
        self.features_len = 1
        self.out_dim = self.final_out_channels+ self.fourier_modes * 2

        # change configs
        self.fraction_order = 0.4