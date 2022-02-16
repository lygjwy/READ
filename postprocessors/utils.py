from .base_postprocessor import BasePostprocessor
from .odin_postprocessor import ODINPostprocessor
from .ebo_postprocessor import EBOPostprocessor
from .reconstruct_postprocessor import ReconstructPostprocessor
from .kld_postprocessor import KldPostprocessor
from .corrupt_reconstruct_postprocessor import CorruptReconstructPostprocessor


def get_postprocessor(name: str, **kwargs):
    post_processors = {
        'msp': BasePostprocessor,
        'odin': ODINPostprocessor,
        'ebo': EBOPostprocessor,
        'reconstruct': ReconstructPostprocessor,
        'kld': KldPostprocessor,
        'corrupt_reconstruct': CorruptReconstructPostprocessor
    }

    return post_processors[name](**kwargs)