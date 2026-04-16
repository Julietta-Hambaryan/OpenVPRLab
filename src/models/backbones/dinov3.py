import torch.nn as nn
from transformers import AutoModel


class DinoV3(nn.Module):
    AVAILABLE_MODELS = [
        'facebook/dinov3-convnext-tiny-pretrain-lvd1689m',
        'facebook/dinov3-convnext-small-pretrain-lvd1689m',
        'facebook/dinov3-convnext-base-pretrain-lvd1689m',
        'facebook/dinov3-convnext-large-pretrain-lvd1689m',
    ]
    DEFAULT_MODEL = 'facebook/dinov3-convnext-base-pretrain-lvd1689m'

    def __init__(
        self,
        backbone_name=DEFAULT_MODEL,
        return_cls_token=False,
    ):
        """DINOv3 ConvNeXt backbone loaded via HuggingFace Transformers.

        Requires: transformers >= 4.56.0

        Args:
            backbone_name (str): HuggingFace model ID. Defaults to ConvNeXt Base.
            return_cls_token (bool): If True, forward() returns (spatial, cls) tuple.
                spatial: (B, C, H//32, W//32)
                cls:     (B, C)  — pooler_output (global avg pool)
                If False, forward() returns only spatial features.
        """
        super().__init__()

        self.backbone_name = backbone_name
        self.return_cls_token = return_cls_token

        if self.backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Backbone '{self.backbone_name}' is not recognised. "
                f"Supported models: {self.AVAILABLE_MODELS}"
            )

        self.dino = AutoModel.from_pretrained(backbone_name)
        self.dino.requires_grad_(False)  # freeze all — inference only

        # Output channels = last ConvNeXt stage dimension (1024 for base)
        self.out_channels = self.dino.config.hidden_sizes[-1]

    def forward(self, x):
        # x: (B, 3, H, W) — ImageNet-normalised
        outputs = self.dino(pixel_values=x)

        # last_hidden_state: (B, C, H//32, W//32) for ConvNeXt — already spatial
        spatial = outputs.last_hidden_state

        if self.return_cls_token:
            # pooler_output: (B, C) — global average pool of last stage
            cls = outputs.pooler_output
            return spatial, cls

        return spatial
