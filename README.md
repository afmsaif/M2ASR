<p align="center">
  <img src="figure/model.pdf" width="300" title="BL-JUST Framework">
</p>



# M2ASR: Multilingual Multi-Task Automatic Speech Recognition via Multi-Objective Optimization

## Abstract
To enable the capability of speech models across multiple languages, training multilingual, multi-task automatic speech recognition (ASR) models has gained growing interest. However, different languages and tasks result in distinct training objectives, potentially leading to conflicts during training and degrading the model’s performance. To overcome this issue, we introduce M2ASR, a multilingual, multi-task ASR framework, which formulates the problem as a constrained multi-objective optimization (MOO), where multilingual multi-task supervised training augmented by speech-to-text translation (S2TT) serve as supervised objectives and are subject to the desired performance of multilingual unsupervised training. We employ MOO techniques to avoid conflicts among multiple linguistic representations and tasks during training. Extensive experiments demonstrate that M2ASR outperforms conventional multilingual ASR models by 28.3% to 38.6% across diverse ASR tasks.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributors](#contributors)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Installation
To get started with M2ASR, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/afmsaif/M2ASR.git
    cd M2ASR
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage
To train and evaluate the M2ASR model, follow these steps:

1. **Prepare your dataset:**
    Ensure that your datasets (Librispeech, AISHELL, CoVoST v2) are downloaded and properly formatted.

2. **Training the model:**
    ```sh
    python train.py --config configs/m2asr_config.json
    ```

3. **Evaluating the model:**
    ```sh
    python evaluate.py --checkpoint path/to/checkpoint
    ```

## Model Architecture
M2ASR employs a conformer backbone with multiple conformer blocks, attention heads, and convolutional layers. The shared backbone is parameterized to handle different languages and tasks. Each task/language-specific head includes a linear layer followed by a softmax layer.

**Network Overview:**
- **Backbone:** 8 conformer blocks, 512 hidden dimensions, 8 attention heads.
- **Task-Specific Heads:** Separate heads for ASR and S2TT tasks per language.

## Datasets
M2ASR is evaluated on the following datasets:
- **Librispeech:** An English dataset consisting of 960 hours of speech.
- **AISHELL v1:** A Chinese dataset consisting of 178 hours of speech.
- **CoVoST v2:** A large-scale multilingual ASR and S2TT corpus covering translations from 21 languages into English and from English into 15 languages.

## Training
M2ASR uses a multi-objective optimization framework to train the model with both supervised and unsupervised objectives. The training process mitigates conflicts among different objectives using a conflict-avoidant update mechanism.

**Training Steps:**
1. Unsupervised training with contrastive predictive coding (CPC) loss.
2. Supervised training with connectionist temporal classification (CTC) loss.
3. Joint optimization of ASR and S2TT objectives using a dynamic weighting method (MGDA).

## Evaluation
The model is evaluated on word error rates (WER) across multiple tasks and languages. Performance metrics include comparisons with baseline models and state-of-the-art techniques.

## Results
M2ASR demonstrates significant improvements over conventional multilingual ASR models. The dynamic weighting method yields WER reductions of 28.3% to 38.6%.

**Example Results:**
- **Chinese:** 6.2% WER (-39.2% reduction)
- **English:** 7.3% WER (-38.1% reduction)

## Contributors
- A F M Saif, Rensselaer Polytechnic Institute
- Lisha Chen, Rensselaer Polytechnic Institute
- Xiaodong Cui, IBM Research AI
- Songtao Lu, IBM Research AI
- Brian Kingsbury, IBM Research AI
- Tianyi Chen, Rensselaer Polytechnic Institute

## Acknowledgements
This work was supported by IBM through the IBM-Rensselaer Future of Computing Research Collaboration.

