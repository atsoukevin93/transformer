class ModelConfig:
    """ base blocks config, params common to all models """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        """
        Prints the settings for the encoder model
        """
        model_name = self.__class__.__name__.removesuffix("ModelConfig")
        settings = f"The {model_name} model has the following configuration:\n"
        for att in dir(self):
            if not (att.startswith("__") and att.endswith("__")):
                settings += f"{att}: {getattr(self,att)}\n"
        return settings
