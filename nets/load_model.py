from constants.model import MODEL_CLASSES, SupportedSupervisedModels


def load_model(config):
    model_name = SupportedSupervisedModels(config["name"].lower())  # Convert to enum
    try:
        # Get the model class using the enum and initialize it
        model_class = MODEL_CLASSES[model_name]
        return model_class(**config["params"])
    except KeyError:
        raise ValueError(f"Unsupported model: {config['name']}")
