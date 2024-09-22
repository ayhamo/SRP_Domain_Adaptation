import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision import transforms

import os
import numpy as np


class Load_Dataset(Dataset):
    def __init__(self, dataset, normalize):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train)
            y_train = torch.from_numpy(y_train).long()


        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        
        if X_train.shape.index(min(X_train.shape[1], X_train.shape[2])) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        self.x_data = X_train
        self.y_data = y_train

        self.num_channels = X_train.shape[1]

        if normalize:
            # Assume datashape: num_samples, num_channels, seq_length
            data_mean = torch.FloatTensor(self.num_channels).fill_(0).tolist()  # assume min= number of channels
            data_std = torch.FloatTensor(self.num_channels).fill_(1).tolist()  # assume min= number of channels
            data_transform = transforms.Normalize(mean=data_mean, std=data_std)
            self.transform = data_transform
        else:
            self.transform = None

        self.len = X_train.shape[0]

    def __getitem__(self, index):
        if self.transform is not None:
            output = self.transform(self.x_data[index].view(self.num_channels, -1, 1))
            self.x_data[index] = output.view(self.x_data[index].shape)

        return self.x_data[index].float(), self.y_data[index].long()

    def __len__(self):
        return self.len


def data_generator(data_path, domain_id, dataset_configs, hparams):
    # loading path
    train_dataset = torch.load(os.path.join(data_path, "train_" + domain_id + ".pt"), weights_only=False)
    full_test_dataset = torch.load(os.path.join(data_path, "test_" + domain_id + ".pt"), weights_only=False)

    # Loading datasets
    train_dataset = Load_Dataset(train_dataset, dataset_configs.normalize)
    full_test_dataset = Load_Dataset(full_test_dataset, dataset_configs.normalize)

    # Split TEST data into train and validation
    val_size = int(len(full_test_dataset) * 0.2)  # 20% for validation
    test_size = len(full_test_dataset) - val_size
    test_dataset, val_dataset = random_split(full_test_dataset, [test_size, val_size])

    # Dataloaders
    batch_size = hparams["batch_size"]
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                             shuffle=False, drop_last=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=dataset_configs.drop_last, num_workers=0)
    
    return train_loader, val_loader, test_loader