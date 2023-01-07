from base.base_model_config import ModelConfig


class DecoderModelConfig(ModelConfig):
    """ base blocks config, params common to all models """
    def __init__(self, **kwargs):
        super(DecoderModelConfig, self).__init__(**kwargs)
