# LLM Fine-Tuning & Quantization Experiments

This repository contains notebooks and resources for fine-tuning Large Language Models (LLMs) and understanding the fundamentals of model quantization. It specifically focuses on efficient fine-tuning techniques using **Unsloth** and educational implementations of quantization algorithms (like NF4).

## 📂 Repository Contents

### 1. 🦙 Llama 3.2 Fine-Tuning
**File:** [`Llama_3-2_3B_Instruct_Fine_Tunning_Using_unsloth.ipynb`](./Llama_3-2_3B_Instruct_Fine_Tunning_Using_unsloth.ipynb)

A complete pipeline for fine-tuning the **Llama 3.2 3B Instruct** model. This notebook utilizes the **Unsloth** library to accelerate training and reduce memory usage.

**Key Features:**
* Loading Llama 3.2 3B (Instruct version) in 4-bit precision.
* Setting up LoRA (Low-Rank Adaptation) adapters.
* Training on ServiceNow-AI/R1-Distill-SFT datasets.
* Inference and saving the fine-tuned model.

### 2. 📉 Quantization Basics
**File:** [`quantization_basics.ipynb`](./quantization_basics.ipynb)

An educational notebook that breaks down how LLM weights are compressed. It includes Python implementations of core concepts such as:
* **Nearest-Neighbor Quantization:** Mapping high-precision weights to a limited set of allowed values.
* **NF4 (NormalFloat 4-bit):** Exploring how Look-Up Tables (LUT) are used to simulate 4-bit quantization for normally distributed weights (QLoRA).

## 🛠️ Tech Stack

* **Python**
* **Unsloth:** For faster and memory-efficient fine-tuning.
* **Hugging Face Transformers & PEFT:** For model loading and adapter management.
* **PyTorch:** Core deep learning framework.
* **NumPy:** For manual implementation of quantization logic.

## 🚀 Getting Started

### Prerequisites
To run the fine-tuning notebook, you will need a GPU-enabled environment (Google Colab T4/L4 or a local machine with NVIDIA GPU).

### Installation
If running locally, install the required dependencies:

```bash
# Install Unsloth (supports Llama-3.2)
pip install unsloth

# Install other requirements
pip install torch transformers peft trl numpy
