# AeroVLA: Autonomous Vision-Language Intelligence for Aerial Scene Understanding

### All project related documents:
- Milestone 1: https://drive.google.com/file/d/1bKL752Mr5uliWQJ6HqFzIn82nINLlny1/view?usp=sharing
- Milestone 1.2: https://drive.google.com/file/d/1QpMnHva6zKykwdxEiuaizQgUOJdY28qe/view?usp=sharing
- Milestone 2: https://drive.google.com/file/d/1BL1pjfrNh8rBrSTng5g_PAzDDCZJQEz5/view?usp=sharing
- Demo Link: https://drive.google.com/file/d/1CcUJ7NE5BBEs09e8NblzlVQW4YW0S70n/view?usp=sharing
- Presentation link: https://canva.link/xw2vq1q0aw5np7f

## Table of Contents
1. Overview
2. Research objective
3. Project repository structure
4. Evaluation of AeroVLA
5. Core architecture
6. Datasets used
7. Major technical contributions
8. Training pipeline
9. Deployment
10. Technologies used
11. Debugging
12. Future scope

## Overview

AeroVLA is a multimodal Vision-Language AI system designed for semantic aerial scene understanding using drone and aerial imagery.

The project evolved from disaster-image interpretation into a fully operational semantic aerial intelligence platform capable of:

- extracting visual embeddings,
- aligning visual and language spaces,
- performing semantic scene understanding,
- generating interpretable aerial intelligence outputs,
- and deploying the system through an interactive real-time web interface.

AeroVLA combines modern Vision-Language Models (VLMs), semantic retrieval systems, and interactive deployment pipelines to create a lightweight autonomous aerial intelligence framework.

## Research Objective

The core objective of AeroVLA is to bridge the gap between:

Visual Perception ↔ Language Understanding

using a custom neural alignment architecture.

The system aims to:

- understand aerial environments,
- interpret semantic spatial patterns,
- analyze drone imagery,
- reduce manual aerial scene interpretation,
- and enable real-time AI-assisted reconnaissance.

## Project Repository Structure

The AeroVLA repository is divided into two major development phases:

### 1. CrisisMMD Research Pipeline 
``` (scripts/)```

The scripts/ folder contains the original AeroVLA multimodal disaster-intelligence pipeline developed during Milestone 1 and Milestone 1.2.

This phase focused on:

- disaster image understanding,
- vision-language alignment,
- report generation,
- multimodal grounding.

### Important Scripts (CrisisMMD)
#### train_alignment.py

Core training pipeline for multimodal embedding alignment between:

- SIGLIP visual encoder
- SmolLM2 language model

Responsible for:

- feature alignment,
- bridge training,
- loss optimization.

#### model_bridge.py

Implements the AeroVLA Neural Bridge.

Architecture:
```
768D Vision Embeddings
→ MLP Projection
→ 576D Language Space
``` 
This became the core innovation layer of AeroVLA.

#### data_loader.py

Custom CrisisMMD dataset loader.

Handles:

- image loading,
- metadata parsing,
- preprocessing,
- batching.

#### scout_inference.py

Inference pipeline for generating disaster reconnaissance reports from unseen images.

#### batch_inference.py

Runs inference over large batches of disaster images and generates structured CSV outputs.

#### global_predictions.py

Stores and manages generated inference outputs during batch evaluation.

#### verify_universal.py

Validation and debugging utility for verifying universal report generation logic.

#### train_phase2.py

Extended alignment experiments and instruction tuning logic.

### 2. Semantic Aerial Intelligence Pipeline (Root Directory)

The root-level files represent the transition into:

AeroVLA Phase 2 — Semantic Aerial Intelligence

This phase focused on:

- aerial scene understanding,
- semantic retrieval,
- VisDrone learning,
- Streamlit deployment,
- live evaluation.

### Important Scripts (VisDrone + Deployment)
#### clip_integration.py

Integrates:

- CLIP ViT-B/32
- semantic embedding extraction
- cosine similarity retrieval

This script established the semantic retrieval pipeline used in final deployment.

#### train_bridge.py

Trains the AeroVLA bridge on VisDrone aerial data.

Training progression:

- 500 image smoke test
- 2000 image scaling
- full 6471-image production training
- Final loss achieved: 0.000013

#### explore_visdrone.py

Dataset inspection and annotation exploration utility.

Used to:

- analyze VisDrone label structures,
- inspect aerial metadata,
- debug annotation parsing.
- visdrone_captioner.py

Experimental semantic caption generation module for aerial imagery.

#### inference_test.py

Primary semantic inference testing script.

Performs:

- image embedding extraction,
- semantic matching,
- confidence scoring,
- MU validation testing.
- app.py

#### Final Streamlit deployment dashboard.

Features:

- image upload,
- semantic scene understanding,
- confidence visualization,
- live evaluation metrics,
- human-in-the-loop validation.

This represents the final AeroVLA deployment interface.

#### aerovla_bridge.py

Final production-ready AeroVLA Neural Bridge implementation used during VisDrone semantic training.

## Evolution of AeroVLA

### Milestone 1 — Disaster Image Understanding

Initial development focused on:

- CrisisMMD disaster dataset,
- SIGLIP visual encoder,
- SmolLM2 language model,
- disaster report generation,
- multimodal embedding alignment.

The system generated reconnaissance-style disaster reports from unseen images.

### Milestone 1.2 — Vision-Language Alignment

Major architectural upgrades included:

- True Vision-Language Pipeline
- inputs_embeds integration
- AeroVLA Neural Bridge
- Structured reconnaissance prompting
- Batch inference over 100+ disaster images
- CSV-based output logging

### Milestone 2 — Semantic Aerial Intelligence

The project transitioned into:

Semantic Retrieval Intelligence

using:

- CLIP ViT-B/32
- VisDrone aerial dataset
- semantic similarity retrieval
- Mahindra University validation data
- Streamlit deployment dashboard

This milestone established AeroVLA as a complete end-to-end aerial semantic intelligence platform.

## Core Architecture

AeroVLA follows the pipeline:

Image → Vision Encoder → Neural Bridge → Semantic Retrieval → Interactive Deployment

## Models Used

### Vision Models
#### SIGLIP

Used during Milestone 1 for disaster understanding tasks.

#### CLIP ViT-B/32

Used during Milestone 2 for semantic aerial retrieval and zero-shot classification.

### Language Model
#### SmolLM2-135M

Used for semantic grounding and early generative experimentation.

## Neural Bridge Architecture

A custom MLP projection module was developed to align:

CLIP visual embeddings (768D)
SmolLM language embeddings (576D)

Architecture:

``` Python
nn.Linear(768, 1024)
nn.GELU()
nn.Linear(1024, 576)
``` 

This bridge became the core innovation layer of AeroVLA.

## Dataset

### CrisisMMD

- CrisisMMD official dataset (version v2.0): https://crisisnlp.qcri.org/crisismmd
- CrisisMMD tar.gz file: https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz
- CrisisMMD dataset via hugging face: https://huggingface.co/datasets/QCRI/CrisisMMD

Contains real disaster images (wildfires, floods, etc.)
Includes metadata and annotations
Used for training and testing

### VisDrone2019-DET

- Offical dataset (DET version): https://github.com/VisDrone/VisDrone-Dataset
- Hugging face link: https://huggingface.co/datasets/Voxel51/VisDrone2019-DET

### Mahindra University Validation Dataset

A custom real-world validation dataset was manually collected from Mahindra University campus.

## Major Technical Contributions
### Vision-Language Embedding Alignment

Implemented a trainable neural bridge between vision-space and language-space.

## Semantic Retrieval System

Transitioned from unstable language generation into:

Zero-Shot Semantic Retrieval Intelligence

using cosine similarity between:

- image embeddings,
- semantic label embeddings.

## Hallucination Analysis

A major research finding of AeroVLA was discovering that:

Low training loss ≠ reliable grounded generation

This led to the architectural pivot toward semantic retrieval systems.

## Interactive Evaluation Framework

The Streamlit deployment includes:

- live semantic inference,
- confidence visualization,
- human-in-the-loop evaluation,
- real-time validation metrics,
- interactive benchmarking.

## Training Pipeline

Training progression:

### Phase 1
500 images
5 epochs
smoke testing

### Phase 2
2000 images
10 epochs
scaling experiment

### Phase 3
Full 6471-image VisDrone dataset
production training

#### Final loss achieved:
#### 0.000013

## Deployment

AeroVLA was deployed through a Streamlit-based dashboard called:

### AeroVLA Mission Control

Features:

- image upload interface,
- semantic intelligence panel,
- top semantic matches,
- confidence visualization,
- technical architecture display,
- live evaluation metrics.

### Evaluation Metrics

The deployment system supports:

- Accuracy
- Confidence Scores
- Correct Predictions
- Incorrect Predictions
- Evaluated Samples

through interactive human validation.

### Sample Semantic Outputs

Examples successfully recognized:

- vehicles parked
- building entrance
- sports field
- trees and vegetation
- recreational area
- parking area

## Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- Streamlit
- CLIP
- SIGLIP
- SmolLM2
- PIL
- Pandas
- NumPy

## Major Debugging Challenges Solved
- Matrix dimension mismatches
- Float32 vs BFloat16 conflicts
- CLIP pooling extraction errors
- Streamlit deployment issues
- Tensor casting failures
- Hallucinated generation outputs
- Remote DGX execution issues
- Git synchronization and backup safety
  
## Deployment Demonstration

A live deployment demonstration video is available here:

- https://drive.google.com/file/d/1CcUJ7NE5BBEs09e8NblzlVQW4YW0S70n/view?usp=sharing

The demo showcases:

- semantic inference,
- confidence retrieval,
- live evaluation,
- MU campus testing,
- and deployment workflow.

## Current Capabilities

AeroVLA currently supports:

- Vision-language alignment
- Semantic aerial scene understanding
- Zero-shot semantic retrieval
- Interactive deployment
- Real-world validation
- Human-in-the-loop evaluation
- DGX remote execution
  
## Future Scope (Milestone 3)

Future planned integrations include:

- MiDaS / ZoeDepth
- YOLO object detection
- Real-time drone streams
- Temporal memory systems
- Multi-frame reasoning
- Video intelligence
- Autonomous navigation reasoning
- Advanced VLM integration
- Public cloud deployment

## Project Status
Milestone 2 — COMPLETED

AeroVLA has successfully evolved into a functioning autonomous semantic aerial intelligence platform.
