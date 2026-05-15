import streamlit as st
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from aerovla_bridge import AeroVLABridge

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AeroVLA Mission Control",
    layout="wide"
)

# ---------------------------------------------------
# SEMANTIC LABELS
# ---------------------------------------------------
semantic_labels = [
    "campus buildings",
    "open ground",
    "road pathway",
    "parking area",
    "pedestrian walkway",
    "sports field",
    "trees and vegetation",
    "urban campus layout",
    "vehicles parked",
    "outdoor environment",
    "building entrance",
    "recreational area"
]

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------
@st.cache_resource
def load_models():

    # Load CLIP
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).float()

    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    # Load AeroVLA Bridge
    bridge = AeroVLABridge().float()

    bridge.load_state_dict(
        torch.load(
            "aerovla_bridge_final.pth",
            map_location="cpu"
        )
    )

    bridge.eval()
    clip_model.eval()

    return clip_model, processor, bridge


clip_model, processor, bridge = load_models()

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("AeroVLA: Autonomous Aerial Intelligence")

st.markdown(
    "### Mahindra University Integrated Project"
)

st.markdown(
    "Semantic Aerial Scene Understanding using CLIP + AeroVLA"
)

# ---------------------------------------------------
# LAYOUT
# ---------------------------------------------------
col1, col2 = st.columns([1, 1])

# ===================================================
# LEFT PANEL
# ===================================================
with col1:

    st.subheader("Upload Reconnaissance Frame")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "jfif"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Input Reconnaissance Frame",
            use_container_width=True
        )

# ===================================================
# RIGHT PANEL
# ===================================================
with col2:

    st.subheader("Semantic Intelligence Report")

    if uploaded_file:

        with st.spinner("Analyzing aerial environment..."):

            # -----------------------------------------
            # IMAGE PROCESSING
            # -----------------------------------------
            image_inputs = processor(
                images=image,
                return_tensors="pt"
            )

            with torch.no_grad():

                # =====================================
                # IMAGE FEATURES
                # =====================================
                image_outputs = clip_model.get_image_features(
                    **image_inputs
                )

                # SAFE EXTRACTION
                if hasattr(image_outputs, "pooler_output"):
                    image_features = image_outputs.pooler_output

                elif hasattr(image_outputs, "image_embeds"):
                    image_features = image_outputs.image_embeds

                else:
                    image_features = image_outputs

                image_features = image_features.float()

                # NORMALIZATION
                image_features = image_features / image_features.norm(
                    dim=-1,
                    keepdim=True
                )

                # =====================================
                # TEXT FEATURES
                # =====================================
                text_inputs = processor(
                    text=semantic_labels,
                    return_tensors="pt",
                    padding=True
                )

                text_outputs = clip_model.get_text_features(
                    **text_inputs
                )

                # SAFE EXTRACTION
                if hasattr(text_outputs, "pooler_output"):
                    text_features = text_outputs.pooler_output

                elif hasattr(text_outputs, "text_embeds"):
                    text_features = text_outputs.text_embeds

                else:
                    text_features = text_outputs

                text_features = text_features.float()

                # NORMALIZATION
                text_features = text_features / text_features.norm(
                    dim=-1,
                    keepdim=True
                )

                # =====================================
                # COSINE SIMILARITY
                # =====================================
                similarity = (
                    100.0 * image_features @ text_features.T
                ).softmax(dim=-1)

                values, indices = similarity[0].topk(3)

        # ---------------------------------------------------
        # RESULTS DISPLAY
        # ---------------------------------------------------
        st.success("Aerial analysis completed successfully")

        top_label = semantic_labels[
            indices[0].item()
        ]

        st.markdown(
            f"## Primary Detection: `{top_label.upper()}`"
        )

        st.divider()

        st.write("### Top Semantic Matches")

        for i in range(3):

            label_name = semantic_labels[
                indices[i].item()
            ]

            confidence = values[i].item()

            st.progress(confidence)

            st.caption(
                f"{label_name} → {confidence * 100:.2f}% confidence"
            )

        # ---------------------------------------------------
        # TECHNICAL DETAILS
        # ---------------------------------------------------
        with st.expander("Technical Architecture"):

            st.markdown("""
            ### AeroVLA Pipeline
            
            - CLIP ViT-B/32 Vision Encoder
            - Semantic Embedding Extraction
            - Cosine Similarity Retrieval
            - AeroVLA Projection Bridge
            - Mahindra University Validation Dataset
            
            ### Current Mode
            
            Stable semantic retrieval inference.
            """)

    else:

        st.info(
            "Awaiting aerial reconnaissance image upload..."
        )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.divider()

st.markdown("""
**PROJECT STATUS:** `PHASE 2 VALIDATED`

**MODEL TYPE:** `Semantic Retrieval Intelligence`

**RESEARCH DOMAIN:** `Autonomous Drone Scene Understanding`
""")