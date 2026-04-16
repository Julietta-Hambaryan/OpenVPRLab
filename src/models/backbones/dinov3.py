import timm
import torch.nn as nn


class DinoV3(nn.Module):

    AVAILABLE_MODELS = {
        'facebook/dinov3-convnext-tiny-pretrain-lvd1689m':  'convnext_tiny.dinov3_in1k',
        'facebook/dinov3-convnext-small-pretrain-lvd1689m': 'convnext_small.dinov3_in1k',
        'facebook/dinov3-convnext-base-pretrain-lvd1689m':  'convnext_base.dinov3_in1k',
        'facebook/dinov3-convnext-large-pretrain-lvd1689m': 'convnext_large.dinov3_in1k',
    }

    DEFAULT_MODEL = 'facebook/dinov3-convnext-base-pretrain-lvd1689m'

    def __init__(
        self,
        backbone_name=DEFAULT_MODEL,
        return_cls_token=False,
    ):
        """DINOv3 ConvNeXt backbone loaded via timm (no HuggingFace gating needed).

        Requires: timm >= 1.0.20

        Args:
            backbone_name (str): HuggingFace-style model ID (same API as before).
            return_cls_token (bool): If True, forward() returns (spatial, cls) tuple.
                spatial: (B, C, H//32, W//32)
                cls:     (B, C) — global average pool of last stage
                If False, forward() returns only spatial features.
        """
        super().__init__()

        self.backbone_name   = backbone_name
        self.return_cls_token = return_cls_token

        if backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Backbone '{backbone_name}' is not recognised. "
                f"Supported models: {list(self.AVAILABLE_MODELS.keys())}"
            )

        timm_name = self.AVAILABLE_MODELS[backbone_name]
        self.model = timm.create_model(timm_name, pretrained=True, features_only=True)
        self.model.requires_grad_(False)  # freeze all — inference only

        # Output channels = last ConvNeXt stage dimension (1024 for base)
        self.out_channels = self.model.feature_info[-1]['num_chs']

    def forward(self, x):
        # x: (B, 3, H, W) — ImageNet-normalised
        features = self.model(x)
        spatial  = features[-1]   # timm ConvNeXt: (B, H, W, C) channels-last

        # Convert to channels-first (B, C, H, W) as expected by all aggregators
        if spatial.shape[-1] == self.out_channels:
            spatial = spatial.permute(0, 3, 1, 2).contiguous()

        if self.return_cls_token:
            cls = spatial.mean(dim=[2, 3])   # (B, C) — global avg pool
            return spatial, cls

        return spatial