import torch
import torch.nn as nn

from modules import parse_config

from config import CONFIG

class ConvolutionalEncoder(nn.Module):
    def __init__(self):
        module_params = CONFIG['model']['convolutional_encoder']
        self.encoder = nn.Sequential(parse_config(module_params))

    def forward(self, x):
        return self.encoder(x)