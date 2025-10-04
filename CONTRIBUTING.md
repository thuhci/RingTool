##  🧱 Contributing
We welcome contributions to RingTool! If you have suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

Some example contributions include:
* Adding new algorithms or models.
* Improving documentation or examples.
* Enhancing performance or usability.
* Fixing bugs or issues.
* Adding new datasets or benchmarks.
* Improving the configuration system.

### Add a new supervised model
1. Create a new file in the [`nets`](nets) directory, e.g., `nets/new_model.py`.
2. Append the new model registration to the [`constants/model.py`](constants/model.py) like below:

```python
from enum import Enum

from nets.inception_time import InceptionTime
from nets.mamba2 import RingToolMamba
from nets.resnet import ResNet1D
from nets.transformer import RingToolBERT

# Import your new model
from nets.new_model import NewModel  


class SupportedSupervisedModels(Enum):
    RESNET = "resnet"
    INCEPTION_TIME = "inception_time"
    TRANSFORMER = "transformer"
    MAMBA2 = "mamba2"
    NEW_MODEL = "new_model"  # Add your new model here


MODEL_CLASSES = {
    SupportedSupervisedModels.RESNET: ResNet1D,
    SupportedSupervisedModels.INCEPTION_TIME: InceptionTime,
    SupportedSupervisedModels.TRANSFORMER: RingToolBERT,
    SupportedSupervisedModels.MAMBA2: RingToolMamba,
    SupportedSupervisedModels.NEW_MODEL: NewModel,  # Add your new model here
}
```
3. Add logic to the [`main.py`](main.py) to use the model in the following training and evaluation process.
4. Add the model to the configuration files in the [`config`](config) directory. You can refer to the existing models for examples.
