import torch
from torch.utils.data import DataLoader

from config import CONFIG

def train(dataset):
    batch_size = CONFIG['train']['batch_size']
    shuffle = CONFIG['train']['shuffle']

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    for i_batch, batch in enumerate(dataloader):
        print(i_batch, batch[0].shape, batch[1].shape)