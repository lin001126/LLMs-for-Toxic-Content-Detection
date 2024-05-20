# LLMs-for-Toxic-Content-Detection

This repository contains the code for data generation and inference to enhance toxic content detection using large language models (LLMs). The data generation code is located in the `datagen` folder, and the inference code is located in the `eval` folder.

## Overview

With the rapid expansion of the internet and the exponential growth of multimedia content, there has been a corresponding increase in inappropriate content such as pornography, abuse, and vulgarity. This repository explores the application of LLMs to improve the detection of toxic content. We utilize techniques such as prompt engineering, data augmentation, and fine-tuning to enhance the performance of LLMs in identifying harmful online content.

## Data Generation

The `datagen` folder contains scripts for generating explanatory datasets and multi-class datasets based on existing binary classification data. By employing advanced data augmentation techniques, we aim to improve the model's understanding and identification of toxic content.

## Inference and Eval

The `eval` folder includes the code for running inference on various datasets. This folder also contains implementations of different inference frameworks to optimize both speed and accuracy. We utilize these frameworks to test the models in various scenarios, including both online and offline modes. Also the evaluation code is in this folder with dataset such as COLD, CHSD, and ToxiCN


## Model

Our fine-tuned Qwen1.5 4B model, which has shown the best performance across all tested datasets, can be found [here](https://huggingface.co/zjj815/Qwen1.5-4B-Chinese-toxic-content-detection). This model is fine-tuned using LLaMA Pro, balancing performance and inference speed effectively.


## Usage

To use this repository, clone the repository and install the required dependencies:

```bash
git clone https://github.com/lin001126/LLMs-for-Toxic-Content-Detection.git
cd LLMs-for-Toxic-Content-Detection
```
You need to install the required dependencies of the corresponding inference framework



