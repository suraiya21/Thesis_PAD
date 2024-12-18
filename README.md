
# Thesis_PAD

This repository supports **Presentation Attack Detection (PAD)** research, featuring tools for data processing, model training, testing, and evaluation.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [Project Structure](#project-structure)
- [License](#license)

## Overview
The **Thesis_PAD** project focuses on developing deep learning methods for detecting presentation attacks using multi-modal datasets. The repository includes all necessary scripts and configurations to support PAD training and evaluation workflows.

## Remote Identity Proofing Process
![Method](1.png)

## Installation
Clone this repository and install required dependencies:

```bash
git clone https://github.com/suraiya21/Thesis_PAD.git
cd Thesis_PAD
conda env create -f env.yaml
conda activate Fmf
```

## Data Preparation
Download the dataset and pretrained model files and organize them as follows:

- **Training Data**: [Download here](https://drive.google.com/file/d/1TSaMmO16vp5mIskk_HH84bj1fuA1_wK1/view?usp=sharing)
- **Model Weights**: [Download here](https://drive.google.com/file/d/1UkhPmaIKXzfA2ToW-oV8t3ogJjRYc3Ch/view?usp=sharing) (Place in the `FMF_Test` directory)

## Usage
### Training
To train the model, use the following script:

```bash
python solver_main.py
```

### Testing
For testing, run:

```bash
cd FMF_Test
python test.py
```

## Project Structure
- **`configs/`**: Configuration files for models and training setups.
- **`datasets/`**: Scripts for dataset processing.
- **`models/`**: Contains model architectures.
- **`solver_main.py`**: Main training script.
- **`utils/`**: Utility functions for data loading, augmentation, and other helpers.

