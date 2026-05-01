import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import os

def run_inference():
    print("Initializing AeroVLA Brain for Inference...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the Brain (SmolLM2)
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct").to(device)
    # Load the Processor
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    
    img_path = os.path.expanduser("~/AeroVLA/data/CrisisMMD/data_image/california_wildfires/10_10_2017/917791044158185473_0.jpg")
    image = Image.open(img_path).convert("RGB")
    
    # Create the prompt
    prompt = "Review the aerial image data. Current Status: Disaster detected. Task: Describe the wildfire damage."
    
    # NEW LOGIC: We process text and images separately to avoid the 'pixel_values' error
    inputs = processor(text=prompt, return_tensors="pt").to(device)
    
    print("Drone is generating reasoning based on mission parameters...")
    
    # We generate text first to prove the 'Brain' is working on the supercomputer
    generated_ids = model.generate(input_ids=inputs.input_ids, max_new_tokens=100)
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\n--- DRONE SCOUT INITIAL LOG ---")
    print(response)
    print("--------------------------------")
    print("NOTE: The brain is currently 'blind' because the bridge isn't active.")
    print("Phase 2 Goal: Connect the pixels to this text output.")

if __name__ == "__main__":
    run_inference()
