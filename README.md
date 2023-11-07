# 🦙 👀 ⛑ DeepDR-LLM: Integrated Image-based Deep Learning and Language Models for Primary Diabetes Care

**DeepDR-LLM** provides a comprehensive solution for primary diabetes care by integrating image-based deep learning with expansive language models. This repository contains code for leveraging the Vision Transformer (ViT) for image processing, coupled with fine-tuned LLaMA models to generate insightful management recommendations for diabetes patients.

## Features

- **Vision Transformer (ViT)** integration for image data processing.
- A robust pipeline to process image predictions and convert them into textual data for input into a large language model.
- Efficient training, made possible with NVIDIA A100 and optimized with Hugging Face's [PEFT](https://github.com/huggingface/peft) and Tim Dettmers' [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).

## Local Setup

1. **Installing Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Modules

### Module 1: Language Model (LLaMA) Integration

Module 1, powered by LLaMA, excels at generating comprehensive diagnostic and therapeutic recommendations. Its design allows seamless integration with outputs from Module 2.

1. **Dataset Building**

   Ensure that your dataset is prepared as per the structure provided in `DeepDR-LLM/Module1/Minimum/example.txt`. Sample format: [{"instruction":"...","input":"...","output":"..."}]. If the "input" requires DR and DME information, Module 2 can predict these.

2. **Training**

   - Execute `DeepDR-LLM/Module1/training/LLM_train.py` for training.
   - Alternatively, use `DeepDR-LLM/Module1/training/run_train.sh` with the appropriate parameters.
   - **Note**: Ensure `llama-7b` weights are downloaded and stored in `DeepDR-LLM/Module1/llama-7b-weights`. If you have pretrained adapters by LoRA, place them in `DeepDR-LLM/Module1/lora-adapter-weights`.

3. **Inference**

   Refer to `DeepDR-LLM/Module1/inference/inference.py`. Ensure to set necessary key arguments.

### Module 2: Image Prediction & Analysis

Module 2 focuses on the prediction and analysis of fundus images.

1. **Dataset Building**

   Two tasks are included: classification and segmentation. Datasets for both are constructed using .txt files. Each line in the file represents an image. For classification, the format is "imagepath classindex". For segmentation, it's "imagepath maskpath", with segmentation labels having a shape of [C,H,W] where C includes the background category.

2. **Training**

   - For classification models: `DeepDR-LLM/Module2/train_cla.py`
   - For segmentation models: `DeepDR-LLM/Module2/train_seg.py`
   - **Note**: Download pretrained `vit-base` weights from ImageNet before training.

3. **Inference**

   Use `DeepDR-LLM/Module2/test.py` for testing. Ensure to load the trained weights into the respective models. The results will be saved accordingly.

