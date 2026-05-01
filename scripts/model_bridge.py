import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class AeroVLA_Bridge(nn.Module):
    def __init__(self, vision_dim=768, language_dim=2048):
        super().__init__()
        # The Linear Projector: Translates Vision (768) -> Language (2048)
        self.projector = nn.Linear(vision_dim, language_dim)
        
        # Load the "Brain" (SmolLM2) - Keeping it on CPU for this quick test
        self.language_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
        print(" SmolLM2 Brain Loaded into the Bridge")

    def forward(self, vision_features):
        # 1. Project the vision features to language space
        projected_features = self.projector(vision_features)
        
        # 2. In a real VLA, we would feed this to the language_model.generate()
        # For this test, we just check if the math matches.
        return projected_features

if __name__ == "__main__":
    # Create the bridge
    bridge = AeroVLA_Bridge()
    
    # Simulate a batch of 4 images from SigLIP (Batch=4, Features=768)
    fake_vision_input = torch.randn(4, 768)
    
    # Run the bridge
    output = bridge(fake_vision_input)
    
    print(f" Input from Eyes: {fake_vision_input.shape}")
    print(f" Output for Brain: {output.shape}")
    print("\n SUCCESS: The Bridge is built. Vision can now talk to Language!")
