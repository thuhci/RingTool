from enum import Enum

from nets.inception_time import InceptionTime
from nets.mamba2 import TaoMamba
from nets.resnet import ResNet1D
from nets.transformer import TaoBERT


class SupportedSupervisedModels(Enum):
    RESNET = "resnet"
    INCEPTION_TIME = "inception_time"
    TRANSFORMER = "transformer"
    MAMBA2 = "mamba2"


MODEL_CLASSES = {
    SupportedSupervisedModels.RESNET: ResNet1D,
    SupportedSupervisedModels.INCEPTION_TIME: InceptionTime,
    SupportedSupervisedModels.TRANSFORMER: TaoBERT,
    SupportedSupervisedModels.MAMBA2: TaoMamba,
}
