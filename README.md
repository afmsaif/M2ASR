# M2ASR-Multilingual-Multi-Task-Automatic-Speech-Recognition-via-Multi-Objective-Optimization
This repository contains the implementation of the Multi-Modal Adaptive Speech Recognition (M$^2$ASR) framework presented in our Interspeech 2024 paper. The codebase includes data preprocessing, model training, evaluation scripts, and all necessary configurations to reproduce the results reported in the paper.

# M$^2$ASR: Multi-Modal Adaptive Speech Recognition

This repository contains the implementation of the Multi-Modal Adaptive Speech Recognition (M$^2$ASR) framework presented in our Interspeech 2024 paper. The codebase includes data preprocessing, model training, evaluation scripts, and all necessary configurations to reproduce the results reported in the paper.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Results](#results)
7. [Contributing](#contributing)
8. [Citation](#citation)
9. [License](#license)

## Introduction
The M$^2$ASR framework integrates multi-modal data to enhance speech recognition performance. It leverages a combination of Librispeech and AISHELL datasets to train robust models capable of handling both English and Chinese speech. This repository includes methods for:
- Data preprocessing and augmentation
- Training using multiple modalities
- Dynamic and static model adaptation
- Evaluation and result visualization

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/m2asr.git
cd m2asr
pip install -r requirements.txt
