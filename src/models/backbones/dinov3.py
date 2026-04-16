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
        super().__init__()
        self.backbone_name    = backbone_name
        self.return_cls_token = return_cls_token

        if self.backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Backbone '{self.backbone_name}' is not recognised. "
                f"Supported models: {self.AVAILABLE_MODELS}"
            )

        self.dino = AutoModel.from_pretrained(backbone_name)
        self.dino.requires_grad_(False)
        self.out_channels = self.dino.config.hidden_sizes[-1]  # 1024 for base

    def forward(self, x):
        outputs = self.dino(pixel_values=x)
        # ConvNeXt: last_hidden_state is (B, H, W, C) — channels last
        spatial = outputs.last_hidden_state

        # Fix channels-last -> channels-first (B, C, H, W)
        if spatial.ndim == 4 and spatial.shape[-1] == self.out_channels:
            spatial = spatial.permute(0, 3, 1, 2).contiguous()

        if self.return_cls_token:
            cls = outputs.pooler_output  # (B, C) — already correct
            return spatial, cls

        return spatial