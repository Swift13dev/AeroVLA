# AeroVLA: Vision-Language AI for Disaster Image Understanding

## Overview

AeroVLA is a multimodal AI system designed to analyze disaster images and generate human-readable situation reports. The goal is to reduce the gap between image collection and decision-making in emergency scenarios.

The system combines computer vision and natural language processing using a custom bridge module that connects visual features to a language model.

## Problem Statement

During disasters, large amounts of visual data are generated through drones, cameras, and satellites. However, understanding these images still requires manual analysis, which is slow and not scalable.

There is a need for an AI system that can automatically:

* Understand disaster scenes
* Interpret visual conditions
* Generate meaningful reports
* Solution Approach

AeroVLA uses a Vision-Language pipeline:

Image → Vision Encoder → Bridge Module → Language Model → Report

* Vision Encoder (SIGLIP): Extracts features from images
* Bridge Module (Custom): Aligns vision features with language space
* Language Model (SmolLM2): Generates textual reports

## Dataset

For Milestone 1, the CrisisMMD dataset was used.

- CrisisMMD official dataset (version v2.0): https://crisisnlp.qcri.org/crisismmd
- CrisisMMD tar.gz file: https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz
- CrisisMMD dataset via hugging face: https://huggingface.co/datasets/QCRI/CrisisMMD

Contains real disaster images (wildfires, floods, etc.)
Includes metadata and annotations
Used for training and testing

### Future work will include:

* VisDrone
* DOTA (aerial datasets)

## Work Completed
* Environment setup and dependency installation
* Dataset download and structure verification
* Metadata exploration and preprocessing
* Custom data loader implementation (CrisisDataset)
* Model integration (SIGLIP + SmolLM2)
* Custom bridge module development
* Training pipeline implemented (train_alignment.py)
* Loss decreased across epochs (learning confirmed)
* Inference pipeline created (scout_inference.py)
* Generated disaster reports from unseen images

## Results
* End-to-end pipeline is functional
* Model successfully generates disaster reports from images
* Training loss decreased, indicating learning
* Output quality improved after debugging and prompt tuning
  
## Challenges
Dataset path and schema mismatches
Missing dependencies
Tokenizer conflicts
Weak initial outputs

All issues were resolved through iterative debugging and improvements.

## Future Scope
* Use aerial datasets (VisDrone, DOTA)
* Real-time drone integration
* Video-based disaster analysis
* Improved language generation
* Web-based deployment

## Technologies Used
* Python
* PyTorch
* Hugging Face Transformers
* SIGLIP (Vision Model)
* SmolLM2 (Language Model)
* Pandas, PIL
* Status

## Dependencies

The following libraries were used in this project:

- torch  
- transformers  
- huggingface_hub  
- pillow (PIL)  
- pandas  
- sentencepiece  
- protobuf  
- numpy  

Install all dependencies using:

```bash
pip install torch transformers huggingface_hub pillow pandas sentencepiece protobuf numpy
```

### Milestone 1 Completed

- Domain research note
- Data pipeline working
- Initial model running with results
