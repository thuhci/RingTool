from nets.inception_time import InceptionTime
from nets.mamba2 import TaoMamba
from nets.resnet import ResNet1D
from nets.transformer import TaoBERT


def load_model(config):
    if config["name"] == "resnet":
        return ResNet1D(**config["params"])
    elif config["name"] == "transformer":
        return TaoBERT(**config["params"])
    elif config["name"] == "inception_time":
        return InceptionTime(**config["params"])
    elif config["name"] == "mamba2":
        return TaoMamba(**config["params"])
