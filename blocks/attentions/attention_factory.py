from .classic_attention import ClassicAttention
from .linformer_attention import LinformerAttention
from .performer_attention import PerformerAttention


class AttentionFactory:
    """
    A simple factory class to instantiate attention layers depending on the provided attention type

    Attributes
    ----------
    CLASSIC : str
        a static attribute which refers to the classic attention
    LINFORMER : str
        a static attribute which refers to the linformer attention
    PERFORMER : str
        a static attribute which refers to the performer attention

    Methods
    -------
    build_attention(config, **kwargs)
        factory method to instantiate different attention type depending on the provided attention type

    """

    CLASSIC: str = "classic"
    LINFORMER: str = "linformer"
    PERFORMER: str = "performer"

    @staticmethod
    def build_attention(config, **kwargs):
        try:
            if config.attention_type == AttentionFactory.CLASSIC:
                return ClassicAttention(config, **kwargs)
            elif config.attention_type == AttentionFactory.LINFORMER:
                return LinformerAttention(config, **kwargs)
            elif config.attention_type == AttentionFactory.PERFORMER:
                return PerformerAttention(config, **kwargs)
            raise AssertionError("The provided attention type is not yet handled")
        except AssertionError as e:
            print(e + f"\n use {AttentionFactory.CLASSIC}, {AttentionFactory.LINFORMER} or {AttentionFactory.PERFORMER}")
