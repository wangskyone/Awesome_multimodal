from transformers import PerceiverModel, PerceiverForMultimodalAutoencoding, PerceiverConfig, Blip2Model
from transformers.models.perceiver.modeling_perceiver import PerceiverImagePreprocessor
import torch

config = PerceiverConfig.from_pretrained("deepmind/vision-perceiver-fourier", cache_dir="/nas_data/WTY/cache")
processor = PerceiverImagePreprocessor(
                    config,
                    position_encoding_type="fourier",
                    fourier_position_encoding_kwargs={
                        "num_bands": 32,
                        "max_resolution": (16, 224, 224),
                        "sine_only": False,
                        "concat_pos": True,
                    },
                    prep_type="patches",
                    spatial_downsample=4,
                    temporal_downsample=1,
                )

imgae = torch.randint(0, 256, (1, 16, 3, 224, 224))

inputs = processor(imgae)
print(inputs[0].shape, inputs[-1].shape)


X = torch.randn(1, 224)

print(X[:, ::4].shape)