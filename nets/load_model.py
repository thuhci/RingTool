import os

from nets.inception_time import InceptionTime
from nets.resnet import ResNet1D
from nets.transformer import MyTransformer


def load_model(config):
    if config["name"] == "resnet":
        return ResNet1D(
            **config["params"]
        )
    elif config['name'] == "transformer":
        return MyTransformer(
            **config["params"]
        )
    elif config['name'] == "inception_time":
        return InceptionTime(
            **config["params"]
        )
