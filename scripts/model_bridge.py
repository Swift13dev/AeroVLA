import torch
import torch.nn as nn

class AeroVLA_Bridge(nn.Module):
    def __init__(self, vision_dim=768, language_dim=576):
        super().__init__()

        # Multi-Layer Projection Network
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, 1024),
            nn.ReLU(),

            nn.Linear(1024, language_dim),

            nn.Dropout(0.1)
        )

        print(" AeroVLA Bridge: True Vision-Language Projection Active")

    def forward(self, vision_features):
        return self.projector(vision_features)


if __name__ == "__main__":
    bridge = AeroVLA_Bridge()

    fake_input = torch.randn(4, 768)

    output = bridge(fake_input)

    print(f"Input Shape : {fake_input.shape}")
    print(f"Output Shape: {output.shape}")
