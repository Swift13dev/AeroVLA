import os
import sys

import torch
import torch.nn.functional as F

from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer
)

# -------------------------------------------------
# Local imports
# -------------------------------------------------
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

import config
from model_bridge import AeroVLA_Bridge


class AeroVLA_Engine:

    def __init__(self):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"\n Initializing AeroVLA Drone Intelligence on {self.device}...\n"
        )

        # -------------------------------------------------
        # LANGUAGE MODEL
        # -------------------------------------------------
        self.brain = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M-Instruct"
        ).to(self.device)

        self.brain_tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M-Instruct"
        )

        # -------------------------------------------------
        # VISION MODEL
        # -------------------------------------------------
        self.vision_processor = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-224"
        )

        self.vision_model = AutoModel.from_pretrained(
            "google/siglip-base-patch16-224"
        ).vision_model.to(self.device)

        self.vision_model.eval()

        # -------------------------------------------------
        # BRIDGE
        # -------------------------------------------------
        self.bridge = AeroVLA_Bridge().to(self.device)

        # -------------------------------------------------
        # LOAD TRAINED WEIGHTS
        # -------------------------------------------------
        if os.path.exists(config.CHECKPOINT_PATH):

            self.bridge.load_state_dict(
                torch.load(
                    config.CHECKPOINT_PATH,
                    map_location=self.device
                )
            )

            print(" Bridge Weights Loaded Successfully")

        else:
            print(" WARNING: No trained bridge weights found")

        self.bridge.eval()

    # =====================================================
    # REPORT GENERATION
    # =====================================================
    def generate_report(self, image_path):

        try:

            # -------------------------------------------------
            # LOAD IMAGE
            # -------------------------------------------------
            image = Image.open(image_path).convert("RGB")

            # -------------------------------------------------
            # PROCESS IMAGE
            # -------------------------------------------------
            vision_inputs = self.vision_processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)

            # -------------------------------------------------
            # EXTRACT SIGLIP FEATURES
            # -------------------------------------------------
            with torch.no_grad():

                vision_outputs = self.vision_model(
                    vision_inputs.pixel_values
                )

                vision_features = vision_outputs.pooler_output

            # -------------------------------------------------
            # BRIDGE PROJECTION
            # -------------------------------------------------
            with torch.no_grad():

                visual_context = self.bridge(
                    vision_features
                )

            # -------------------------------------------------
            # NORMALIZE VISUAL EMBEDDINGS
            # -------------------------------------------------
            visual_context = F.normalize(
                visual_context,
                dim=-1
            )

            # Reduce visual dominance
            visual_context = visual_context * 0.05

            # -------------------------------------------------
            # PROMPT
            # -------------------------------------------------
            prompt = config.PROMPT_TEMPLATE

            text_inputs = self.brain_tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self.device)

            # -------------------------------------------------
            # TEXT EMBEDDINGS
            # -------------------------------------------------
            text_embeddings = self.brain.get_input_embeddings()(
                text_inputs.input_ids
            )

            # -------------------------------------------------
            # CREATE VISUAL TOKEN
            # -------------------------------------------------
            visual_token = visual_context.unsqueeze(1)

            # Match dtype with LM embeddings
            visual_token = visual_token.to(
                text_embeddings.dtype
            )

            # -------------------------------------------------
            # MERGE TEXT + VISUAL
            # -------------------------------------------------
            combined_embeddings = torch.cat(
                [text_embeddings, visual_token],
                dim=1
            )

            # -------------------------------------------------
            # ATTENTION MASK
            # -------------------------------------------------
            visual_attention = torch.ones(
                (1, 1),
                dtype=torch.long,
                device=self.device
            )

            combined_attention = torch.cat(
                [
                    text_inputs.attention_mask,
                    visual_attention
                ],
                dim=1
            )

            # -------------------------------------------------
            # GENERATE OUTPUT
            # -------------------------------------------------
            with torch.no_grad():

                output = self.brain.generate(
                    inputs_embeds=combined_embeddings,
                    attention_mask=combined_attention,

                    max_new_tokens=config.GEN_CONFIG[
                        "max_new_tokens"
                    ],

                    repetition_penalty=config.GEN_CONFIG[
                        "repetition_penalty"
                    ],

                    do_sample=config.GEN_CONFIG[
                        "do_sample"
                    ],

                    pad_token_id=self.brain_tokenizer.eos_token_id,
                    eos_token_id=self.brain_tokenizer.eos_token_id
                )

            # -------------------------------------------------
            # DECODE OUTPUT
            # -------------------------------------------------
            report = self.brain_tokenizer.decode(
                output[0],
                skip_special_tokens=True
            )

            return report.strip()

        except Exception as e:

            return f"DRONE_SYSTEM_ERROR: {str(e)}"


# =========================================================
# TEST BLOCK
# =========================================================
if __name__ == "__main__":

    engine = AeroVLA_Engine()

    test_img = os.path.join(
        config.DATASET_ROOT,
        "california_wildfires/10_10_2017/917791044158185473_0.jpg"
    )

    if os.path.exists(test_img):

        print("\n--- DRONE RECON TEST ---\n")

        report = engine.generate_report(test_img)

        print(f"REPORT:\n{report}")

    else:

        print(
            f" Test image not found:\n{test_img}"
        )
