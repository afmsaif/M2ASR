# M2ASR: Multilingual Multi-Task Automatic Speech Recognition via Multi-Objective Optimization

This repository contains the code for the paper titled **"M2ASR: Multilingual Multi-Task Automatic Speech Recognition via Multi-Objective Optimization"** by A F M Saif, Lisha Chen, Xiaodong Cui, Songtao Lu, Brian Kingsbury, and Tianyi Chen. The paper was presented at Interspeech 2024.

## Abstract

To enable the capability of speech models across multiple languages, training multilingual, multi-task automatic speech recognition (ASR) models has gained growing interest. However, different languages and tasks result in distinct training objectives, potentially leading to sub-optimal solutions. This work introduces a novel framework for multilingual multi-task ASR using multi-objective optimization to effectively balance the various training objectives and improve overall performance.

## Repository Structure

- `src/` : Contains the source code for the M2ASR model.
- `data/` : Includes scripts for data preprocessing and loading.
- `scripts/` : Contains various utility scripts for training, evaluation, and other tasks.
- `models/` : Pretrained models and model checkpoints.
- `results/` : Directory to save evaluation results and logs.
- `README.md` : This readme file.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone (https://github.com/afmsaif/M2ASR.git)
cd m2asr
pip install -r requirements.txt
```

## Acknowledgement

This work was supported by the Rensselaer-IBM AI Research Collaboration, part of the IBM AI Horizons Network
