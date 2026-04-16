import torch
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

    def __init__(self, backbone_name=DEFAULT_MODEL, return_cls_token=False):
        super().__init__()
        self.backbone_name    = backbone_name
        self.return_cls_token = return_cls_token
        self.dino = AutoModel.from_pretrained(backbone_name)
        self.dino.requires_grad_(False)
        self.out_channels = self.dino.config.hidden_sizes[-1]
        
        # Detect actual H, W by doing a dummy forward
        with torch.no_grad():
            _dummy = torch.randn(1, 3, 224, 224)
            _out = self.dino(pixel_values=_dummy).last_hidden_state
            _, N, _ = _out.shape
            # get H, W from model config spatial_dims if available, else find factors
            self._spatial_H, self._spatial_W = self._find_hw(N, 224)
            print(f"DinoV3 spatial grid: {self._spatial_H}x{self._spatial_W} (N={N})")

    @staticmethod
    def _find_hw(N, input_size):
        # find H, W such that H*W=N and H<=W
        for h in range(int(N**0.5), 0, -1):
            if N % h == 0:
                return h, N // h
        return 1, N

    def forward(self, x):
        outputs = self.dino(pixel_values=x)
        spatial = outputs.last_hidden_state  # (B, N, C)

        B, N, C = spatial.shape
        
        # Recompute H, W for this input size in case it differs from init
        if x.shape[2] != 224:
            H, W = self._find_hw(N, x.shape[2])
        else:
            H, W = self._spatial_H, self._spatial_W

        spatial = spatial.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        if self.return_cls_token:
            cls = outputs.pooler_output  # (B, C)
            return spatial, cls

        return spatial