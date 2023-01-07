from base.base_model_config import ModelConfig


class TransformerModelConfig(ModelConfig):
    """ base blocks config, params common to all models """
    def __init__(self, **kwargs):
        super(TransformerModelConfig, self).__init__(**kwargs)