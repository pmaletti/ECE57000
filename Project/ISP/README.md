<div align="center">

# Isometric Spectral Pruning:<br> Structured Pruning of Diffusion Models

<h3>ECE 57000: Artificial Intelligence - Final Project</h3>


**Prajna G. Malettira**<sup>1</sup><br>
<sup>1</sup>Purdue University

</div>

<div align="left">
  <img src="./figures/teasers.png" width="100%" />
  <br>
  <em>Qualitative comparison of structured pruning methods on the SD3.5-Large model. We evaluate OBS-Diff, and our method (ISP) at various sparsity levels (15%, 20%, 30% and 40%) using the same prompt and negative prompt. All images are generated at a resolution of 1024 x 1024.</em>
</div>

## 1. Installation
First, unzip our codebase:

```bash
unzip ISP.zip -d /path_to_home_directory/ISP/
cd ISP
```
Then, install our environment:
```bash
conda env create -f env.yml
```
You need to have access to the models (SDXL and SD-3.5 Large) from [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [SD3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) and calibration dataset (GCC3M) from [Conceptual Caption 12M](https://ai.google.com/research/ConceptualCaptions/download). *(Note: Automatic downloading via scripts is not fully feasible without user authentication/Hugging Face access tokens for gated models).*

## 2. Data Preprocessing
To prepare the calibration dataset, we utilize the GCC3M subset. 
Run the following script to format the raw data for the OBS-Diff pruning pipeline:
```bash
python data/gcc3m_preprocessing.py
```
This step ensures the data is correctly tokenized and organized to facilitate accurate Hessian information estimation during the pruning process.

## 3. Usage

+ **Structured Pruning**
```bash
bash ./scripts/prune_Structured.sh
```

+ **Run Eval**
```bash
bash ./scripts/eval_prune.sh
```
> **Note:** You need to change the path to the models and calibration dataset in the scripts and codes.

## 4. Code Structure
The overall framework architecture is adapted from the **OBS-Diff** repository. The repository is structured as follows:

```text
ISP/
├── data/
│   └── gcc3m_preprocessing.py
├── figures/
│   └── teasers.png
├── lib/
│   ├── dataloader.py
│   ├── IsometricDiffusionPruner_Structured_.py
│   ├── ISP_prune.py
│   ├── OBS_Diff_Structured.py
│   └── prune.py
├── scripts/
│   ├── eval_prune.sh
│   └── prune_Structured.sh
├── .gitignore
├── env.yml
├── main.py
└── README.md
```

## 5. Authorship and Code Attribution
This project builds upon the open-source **OBS-Diff** framework. To implement our geometric saliency criterion, the following modifications and original contributions were made:

*   **Adapted/Copied Code:** The general repository structure, data preprocessing (`data/gcc3m_preprocessing.py`), baseline evaluation scripts, and `lib/OBS_Diff_Structured.py` are adapted directly from the original OBS-Diff repository. 
*   **Original/Rewritten Code (Ours):** 
    *   `lib/ISP_prune.py`: Completely rewritten to replace the inverse Hessian calculation with our spectral truncation method.
    *   `lib/IsometricDiffusionPruner_Structured_.py`: Completely rewritten to implement the local spectral isometry scoring and globally ranked non-uniform sparsity allocation.
    *   `main.py`: Modified lines [INSERT LINE NUMBERS HERE] to initialize ISP instead of OBS-Diff.
    *   `lib/dataloader.py`: Modified lines [INSERT LINE NUMBERS HERE] to support our specific evaluation pipeline.
    *   `scripts/prune_Structured.sh` & `scripts/eval_prune.sh`: Modified lines [INSERT LINE NUMBERS HERE] to point to the new ISP execution paths.