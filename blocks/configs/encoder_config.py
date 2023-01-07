from base.base_model_config import ModelConfig


class EncoderModelConfig(ModelConfig):
    """ base blocks config, params common to all models """
    def __init__(self, **kwargs):
        super(EncoderModelConfig, self).__init__(**kwargs)

