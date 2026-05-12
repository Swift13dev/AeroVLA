import os

# =========================================================
# PATH CONFIGURATION
# =========================================================

DATASET_ROOT = os.path.expanduser(
    "~/AeroVLA/data/CrisisMMD/data_image"
)

OUTPUT_DIR = os.path.expanduser(
    "~/AeroVLA/outputs"
)

MODEL_DIR = os.path.expanduser(
    "~/AeroVLA/models"
)

CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    "universal_bridge.pt"
)

# =========================================================
# BATCH SETTINGS
# =========================================================

IMAGES_PER_FOLDER = 15

CATEGORIES = [
    'california_wildfires',
    'hurricane_harvey',
    'srilanka_floods',
    'mexico_earthquake'
]

# =========================================================
# GENERATION SETTINGS
# =========================================================

GEN_CONFIG = {

    # Slightly longer reports
    "max_new_tokens": 80,

    # Reduces repetitive hallucinations
    "repetition_penalty": 1.4,

    # Deterministic generation
    "do_sample": False,
}

# =========================================================
# DRONE OBSERVATION PROMPT
# =========================================================

PROMPT_TEMPLATE = """
[DRONE AERIAL OBSERVATION REPORT]

Observe ONLY visible details from the aerial disaster image.

STRICT RULES:
- Do NOT explain causes
- Do NOT mention human intentions
- Do NOT assume timelines
- Do NOT invent events
- Do NOT generate fictional details
- Report ONLY visible observations

Focus ONLY on:
- Smoke
- Fire
- Flood water
- Buildings
- Roads
- Vegetation
- Debris
- Structural damage

Required Format:

Visible Elements:
-

Scene Analysis:
-

Risk Level:
-
"""
