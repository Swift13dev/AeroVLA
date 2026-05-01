
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import os
from model_bridge import AeroVLA_Bridge

def run_scout_report():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing AeroVLA Dual-Stream System on {device}...")

    # 1. Load Brain, Eyes, and the specific Brain Tokenizer
    brain = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct").to(device)
    brain_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    vision_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    bridge = AeroVLA_Bridge().to(device)

    # 2. Image Selection
    img_path = os.path.expanduser("~/AeroVLA/data/CrisisMMD/data_image/california_wildfires/10_10_2017/917791044158185473_0.jpg")
    image = Image.open(img_path).convert("RGB")

    # List of images to test
    test_images = [
        "california_wildfires/10_10_2017/917791044158185473_0.jpg",
        "california_wildfires/10_10_2017/917791291823591425_1.jpg"
    ]
    
    print("\n--- SELECT IMAGE FOR SCOUT RECON ---")
    for i, path in enumerate(test_images):
        print(f"[{i}] {path.split('/')[-1]}")
    
    choice = int(input("\nEnter image index [0 or 1]: "))
    img_rel_path = test_images[choice]
    
    img_path = os.path.expanduser(f"~/AeroVLA/data/CrisisMMD/data_image/{img_rel_path}")

    # 3. Prompt
    prompt = "<|im_start|>system\nYou are a Disaster Response Scout Drone. Report findings.<|im_end|>\n<|im_start|>user\nAnalyze this wildfire image.<|im_end|>\n<|im_start|>assistant\n"

    # 4. Processing (Split streams)
    text_inputs = brain_tokenizer(prompt, return_tensors="pt").to(device)
    vision_inputs = vision_processor(images=image, return_tensors="pt").to(device)

    print("Drone is generating autonomous reasoning...")
    
    # 5. Generate
    output = brain.generate(
        input_ids=text_inputs.input_ids,
        attention_mask=text_inputs.attention_mask,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.8,
        repetition_penalty=1.5,
        pad_token_id=brain_tokenizer.eos_token_id
    )

    # 6. Decode using the CORRECT tokenizer
    report = brain_tokenizer.decode(output[0], skip_special_tokens=True)
    
    print("\n" + "="*30)
    print("EROVLA MISSION LOG")
    print("="*30)
    print(f"REPORT: {report.split('assistant')[-1].strip()}")
    print("="*30)

if __name__ == "__main__":
    run_scout_report()
