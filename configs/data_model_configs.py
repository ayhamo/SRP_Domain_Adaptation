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

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128


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
        # TCN features
        self.tcn_layers = [32,64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 15 # 25
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100


class WISDM(object):
    def __init__(self):
        super(WISDM, self).__init__()
        self.class_names = ['walk', 'jog', 'sit', 'stand', 'upstairs', 'downstairs']
        self.sequence_len = 128
        # Closed Set DA
        self.scenarios = [("2", "32"), ("4", "15"),("7", "30"),('12','7'), ('12','19'),('18','20'),\
                          ('20','30'), ("21", "31"),("25", "29"), ('26','2')]
        
        # H score scnearios
        #self.scenarios = [('0', '11'), ('0', '12'), ('0', '17'), ('0', '18'), ('0', '19'), ('0', '2'), ('0', '20'), ('0', '23'), ('0', '26'), ('0', '27'), ('0', '28'), ('0', '29'), ('0', '3'), ('0', '30'), ('0', '31'), ('0', '32'), ('0', '33'), ('0', '34'), ('0', '35'), ('0', '4'), ('0', '5'), ('0', '6'), ('0', '7'), ('0', '9'), ('1', '0'), ('1', '10'), ('1', '11'), ('1', '12'), ('1', '13'), ('1', '14'), ('1', '15'), ('1', '16'), ('1', '17'), ('1', '18'), ('1', '19'), ('1', '2'), ('1', '20'), ('1', '21'), ('1', '22'), ('1', '23'), ('1', '25'), ('1', '26'), ('1', '27'), ('1', '28'), ('1', '3'), ('1', '30'), ('1', '31'), ('1', '32'), ('1', '33'), ('1', '34'), ('1', '35'), ('1', '4'), ('1', '5'), ('1', '6'), ('1', '7'), ('1', '9'),('8', '0'), ('8', '11'), ('8', '18'), ('8', '7'), ('9', '11'), ('9', '12'), ('9', '17'), ('9', '18'), ('9', '19'), ('9', '2'), ('9', '20'), ('9', '23'), ('9', '26'), ('9', '28'), ('9', '29'), ('9', '3'), ('9', '30'), ('9', '31'), ('9', '32'), ('9', '33'), ('9', '34'), ('9', '35'), ('9', '4'), ('9', '5'), ('9', '6'), ('9', '7'),('10', '11'), ('10', '12'), ('10', '17'), ('10', '18'), ('10', '19'), ('10', '2'), ('10', '20'), ('10', '23'), ('10', '26'), ('10', '27'), ('10', '28'), ('10', '29'), ('10', '3'), ('10', '30'), ('10', '31'), ('10', '32'), ('10', '33'), ('10', '34'), ('10', '35'), ('10', '4'), ('10', '5'), ('10', '6'), ('10', '7'), ('10', '9'), ('13', '11'), ('13', '12'), ('13', '15'), ('13', '17'), ('13', '18'), ('13', '19'), ('13', '2'), ('13', '20'), ('13', '23'), ('13', '26'), ('13', '27'), ('13', '28'), ('13', '3'), ('13', '30'), ('13', '31'), ('13', '32'), ('13', '33'), ('13', '34'), ('13', '35'), ('13', '4'), ('13', '5'), ('13', '6'), ('13', '7'), ('13', '9'), ('14', '11'), ('14', '12'), ('14', '17'), ('14', '18'), ('14', '19'), ('14', '2'), ('14', '20'), ('14', '23'), ('14', '26'), ('14', '27'), ('14', '28'), ('14', '29'), ('14', '3'), ('14', '30'), ('14', '31'), ('14', '32'), ('14', '33'), ('14', '34'), ('14', '35'), ('14', '4'), ('14', '5'), ('14', '6'), ('14', '7'), ('14', '9'), ('15', '0'), ('15', '1'), ('15', '10'), ('15', '11'), ('15', '12'), ('15', '13'), ('15', '14'), ('15', '16'), ('15', '17'), ('15', '18'), ('15', '19'), ('15', '2'), ('15', '20'), ('15', '21'), ('15', '22'), ('15', '23'), ('15', '25'), ('15', '26'), ('15', '28'), ('15', '3'), ('15', '30'), ('15', '31'), ('15', '32'), ('15', '33'), ('15', '34'), ('15', '35'), ('15', '4'), ('15', '5'), ('15', '6'), ('15', '7'), ('15', '9'), ('16', '11'), ('16', '12'), ('16', '15'), ('16', '17'), ('16', '18'), ('16', '19'), ('16', '2'), ('16', '20'), ('16', '23'), ('16', '26'), ('16', '27'), ('16', '28'), ('16', '29'), ('16', '3'), ('16', '30'), ('16', '31'), ('16', '32'), ('16', '33'), ('16', '34'), ('16', '35'), ('16', '4'), ('16', '5'), ('16', '6'), ('16', '7'), ('16', '9'), ('21', '11'), ('21', '12'), ('21', '15'), ('21', '17'), ('21', '18'), ('21', '19'), ('21', '2'), ('21', '20'), ('21', '23'), ('21', '26'), ('21', '27'), ('21', '28'), ('21', '29'), ('21', '3'), ('21', '30'), ('21', '31'), ('21', '32'), ('21', '33'), ('21', '34'), ('21', '35'), ('21', '4'), ('21', '5'), ('21', '6'), ('21', '7'), ('21', '9'), ('22', '11'), ('22', '12'), ('22', '15'), ('22', '17'), ('22', '18'), ('22', '19'), ('22', '2'), ('22', '20'), ('22', '23'), ('22', '26'), ('22', '27'), ('22', '28'), ('22', '29'), ('22', '3'), ('22', '30'), ('22', '31'), ('22', '32'), ('22', '33'), ('22', '34'), ('22', '35'), ('22', '4'), ('22', '5'), ('22', '6'), ('22', '7'), ('22', '9'), ('24', '0'), ('24', '10'), ('24', '11'), ('24', '12'), ('24', '13'), ('24', '14'), ('24', '16'), ('24', '17'), ('24', '18'), ('24', '19'), ('24', '2'), ('24', '20'), ('24', '21'), ('24', '22'), ('24', '23'), ('24', '25'), ('24', '26'), ('24', '28'), ('24', '3'), ('24', '30'), ('24', '31'), ('24', '32'), ('24', '33'), ('24', '34'), ('24', '35'), ('24', '4'), ('24', '5'), ('24', '6'), ('24', '7'), ('24', '9'), ('25', '11'), ('25', '12'), ('25', '15'), ('25', '17'), ('25', '18'), ('25', '19'), ('25', '2'), ('25', '20'), ('25', '23'), ('25', '26'), ('25', '27'), ('25', '28'), ('25', '29'), ('25', '3'), ('25', '30'), ('25', '31'), ('25', '32'), ('25', '33'), ('25', '34'), ('25', '35'), ('25', '4'), ('25', '5'), ('25', '6'), ('25', '7'), ('25', '9'), ('27', '0'), ('27', '1'), ('27', '10'), ('27', '11'), ('27', '12'), ('27', '13'), ('27', '14'), ('27', '15'), ('27', '16'), ('27', '17'), ('27', '18'), ('27', '19'), ('27', '2'), ('27', '20'), ('27', '21'), ('27', '22'), ('27', '23'), ('27', '25'), ('27', '26'), ('27', '28'), ('27', '29'), ('27', '3'), ('27', '30'), ('27', '31'), ('27', '32'), ('27', '33'), ('27', '34'), ('27', '35'), ('27', '4'), ('27', '5'), ('27', '6'), ('27', '7'), ('27', '9'), ('29', '0'), ('29', '1'), ('29', '10'), ('29', '11'), ('29', '12'), ('29', '13'), ('29', '14'), ('29', '16'), ('29', '17'), ('29', '18'), ('29', '19'), ('29', '2'), ('29', '20'), ('29', '21'), ('29', '22'), ('29', '23'), ('29', '25'), ('29', '26'), ('29', '28'), ('29', '3'), ('29', '30'), ('29', '31'), ('29', '32'), ('29', '33'), ('29', '34'), ('29', '35'), ('29', '4'), ('29', '5'), ('29', '6'), ('29', '7'), ('29', '9'), ('3', '11'), ('3', '12'), ('3', '15'), ('3', '17'), ('3', '18'), ('3', '19'), ('3', '2'), ('3', '20'), ('3', '23'), ('3', '26'), ('3', '27'), ('3', '28'), ('3', '29'), ('3', '30'), ('3', '31'), ('3', '32'), ('3', '33'), ('3', '34'), ('3', '35'), ('3', '4'), ('3', '5'), ('3', '6'), ('3', '7'), ('3', '9'), ('34', '0'), ('34', '10'), ('34', '11'), ('34', '12'), ('34', '13'), ('34', '14'), ('34', '15'), ('34', '16'), ('34', '17'), ('34', '18'), ('34', '19'), ('34', '2'), ('34', '20'), ('34', '21'), ('34', '22'), ('34', '23'), ('34', '25'), ('34', '26'), ('34', '27'), ('34', '28'), ('34', '29'), ('34', '3'), ('34', '30'), ('34', '31'), ('34', '32'), ('34', '33'), ('34', '35'), ('34', '4'), ('34', '5'), ('34', '6'), ('34', '7'), ('34', '9')]

        self.num_classes = 6
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6
        self.width = 64  # for FNN
        self.fourier_modes = 64
        # features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.out_dim = self.final_out_channels+ self.fourier_modes * 2
        # For DANCE UNI
        #self.out_dim = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75,150,300]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500




class HHAR_SA(object):  ## HHAR dataset, SAMSUNG device.
    def __init__(self):
        super(HHAR_SA, self).__init__()
        self.sequence_len = 128
        #self.scenarios = [("0", "2"), ("2","4")]
        self.scenarios = [("0", "2"), ("1", "6"),("2", "4"),("4", "0"),("4", "5"),\
            ("5", "1"),("5", "2"),("7", "2"),("7", "5"),("8", "4")]
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
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

        # TCN features
        self.tcn_layers = [75,150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500

class Boiler(object):
    def __init__(self):
        super(Boiler, self).__init__()
        self.class_names = ['0','1']
        self.sequence_len = 6
        self.scenarios = [("1", "2"),("1", "3"),("2", "3")]
        self.num_classes = 2
        self.sequence_len = 6
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 20
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.2

        # features
        self.mid_channels = 32
        self.final_out_channels = 64
        self.features_len = 1

        # TCN features
        self.tcn_layers = [32,64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 15# 25
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100