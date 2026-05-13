import torch.nn as nn

class AeroVLABridge(nn.Module):
    def __init__(self, clip_dim=768, smollm_dim=576):
        super(AeroVLABridge, self).__init__()
        # Projects CLIP 768-D visual features to SmolLM2 576-D language space
        self.projector = nn.Sequential(
            nn.Linear(clip_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, smollm_dim)
        )

    def forward(self, x):
        return self.projector(x)