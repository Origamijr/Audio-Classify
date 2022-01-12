import torch.nn as nn
import inspect


class LayerFactory:
    """
    Class for singular layer generation from a parameter dictionary
    """
    def __init__(self, layer_fun, **kwargs):
        self.f = layer_fun
        self.kwargs = kwargs
        self.make = self._make

        # filter out only the kwargs needed for the layer function
        """
        sig = inspect.signature(self.f)
        filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
        self.f_kwargs = {filter_key: self.kwargs[filter_key] for filter_key in filter_keys if filter_key in self.kwargs}
        """
        self.f_kwargs = {key: self.kwargs[key] for key in self.kwargs if key not in ['type', 'repeat']}

    class InvalidLayer(Exception):
        pass

    @staticmethod
    def make(conf):
        """
        Static instatiator for a single layer
        """
        type = conf['type']
        if type == 'linear':
            f = nn.Linear
        elif type == 'conv2d':
            f = nn.Conv2d
        elif type == 'gru':
            f = nn.GRU
        elif type == 'relu':
            f = nn.ReLU
        elif type == 'tanh':
            f = nn.Tanh
        elif type == 'flatten':
            f = nn.Flatten
        else:
            raise LayerFactory.InvalidLayer(type)
        return LayerFactory(f, **conf).make()

    def _make(self):
        """
        Outputs a layer
        "Overrides" static make when called on an instance
        """
        layer = self.f(**self.f_kwargs)
        # TODO handle weight initialization here
        return layer


class ResidualCell(nn.Module):
    """
    Simple pre-activation residual cell with no batch normalization
    """
    def __init__(self, module_params):
        super(ResidualCell, self).__init__()
        layers = parse_config(module_params)
        self.cell = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.cell(x)


def parse_config(module_params):
    """
    Builds a list of pytorch layers from a list of layer parameters (parsed from a configuration file)
    """
    layers = []
    for layer_params in module_params:
        type = layer_params['type']
        repeat = 1 if 'repeat' not in layer_params else layer_params['repeat']
        for i in range(repeat):
            if type == 'sequential': # untested type
                layers.append(nn.Sequential(*parse_config(layer_params['cell'])))
            if type == 'residual':
                layers.append(ResidualCell(layer_params['cell']))
            else:
                layers.append(LayerFactory.make(layer_params))
    return layers
