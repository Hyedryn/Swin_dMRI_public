# Leveraging swin transformer for enhanced diagnosis of Alzheimer’s disease using multi-shell diffusion MRI 

This repository provides a deep learning framework for classifying Alzheimer's disease using diffusion MRI data, leveraging microstructural feature maps and a pretrained Swin Transformer.

---

## Features

### Model architectures
- **Implemented models**:  
  - **`SwinTST`**: A Swin Transformer adapted for classification tasks.  
  - **`ResNet`**: ResNet-18 and ResNet-34 architectures for comparison.  

- **Freezing strategies**:  
  - **`allButStage4`**: Freeze all layers except the last stage and the classification head.  
  - **`all`**: Freeze all backbone layers, keeping only the classification head trainable.  
  - **`LoRA`**: Low-Rank Adaptation of Transformers with parameters:
    - `LoRA+r<lora_r>+a<lora_alpha>+d<lora_dropout>+attn/attnmlp`  
    - Example: `LoRA+r4+a1+d0.25+attn` (rank = 4, alpha = 1, dropout = 0.25, applied to attention layers).


### Dynamic data augmentation
- Provides augmentation techniques.
- **Augmentation modes**:  
  - **`F`**: No augmentation.  
  - **`T+t<translation_xyz>+r<rotation>+n<noise>`**:  
    - Example: `T+t555+r0+n0.05` for:
      - Translation: 5 voxels along x, y, and z axes.
      - Rotation: 0°.
      - Noise: 0.05 level Gaussian noise.

---

## Installation

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/Hyedryn/Swin_dMRI_public.git
    cd Swin_dMRI_public
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Dataset

This framework requires diffusion MRI data preprocessed into microstructural feature maps (e.g., NODDI or DTI).

### Input files
1. **Microstructural feature maps**: `.pickle` files containing 3D numpy arrays stored in the format:  
   `dic[subject][session][<microstructural_map>]`.

   - For **DTI** data, `<microstructural_map>` should include:
     - `FA`: Fractional Anisotropy (`sub-{subject}_ses-{session}_FA`).
     - `AD`: Axial Diffusivity (`sub-{subject}_ses-{session}_AD`).
     - `RD`: Radial Diffusivity (`sub-{subject}_ses-{session}_RD`).

   - For **NODDI** data, `<microstructural_map>` should include:
     - `ODI`: Orientation Dispersion Index (`sub-{subject}_ses-{session}_odi`).
     - `fintra`: Fractional Intra-cellular Volume (`sub-{subject}_ses-{session}_fintra`).
     - `fextra`: Fractional Extra-cellular Volume (`sub-{subject}_ses-{session}_fextra`).

2. **Phenotype metadata**: CSV file (`dMRI_phenotypes.csv` or `amyloid_phenotypes.csv`) containing:
   - `sub`: Subject ID.
   - `ses`: Session ID.
   - `Group`: Classification label (`CN`: Control, `MCI`: Mild Cognitive Impairment, `AD`: Alzheimer’s Disease).
   - `AMYLOID_STATUS` (optional): Binary amyloid status (0 or 1).

---

## Usage

### Training the model
Use the `train.py` script to train the model:

```bash
python train.py \
    --data_path <path_to_data> \
    --output_path <path_to_output> \
    --input_details <input_file> <input_type> \
    --model swint \
    --control_variable CN \
    --target_variable MCI AD \
    --batch_size 32 \
    --learning_rate 1e-1 \
    --dynamic_data_augmentation T+t555+r0+n0 \
    --model_freeze LoRA+r4+a1+d0.25+attn \
    --gpu_id 0 \
    --verbose
```

### Key arguments
- **`--data_path`**: Path to the dataset directory.
- **`--output_path`**: Directory to save results and model checkpoints.
- **`--input_details`**: Specifies the input file and type (e.g., `noddi` or `DTI`).
- **`--model`**: Model architecture to use (`swint`, `resnet18`, or `resnet34`).
- **`--control_variable`**: Group used as the control (e.g., `CN`).
- **`--target_variable`**: Groups to classify (e.g., `MCI`, `AD`).
- **`--batch_size`**: Batch size for training.
- **`--learning_rate`**: Initial learning rate for optimization.
- **`--dynamic_data_augmentation`**: Specifies the data augmentation method.
- **`--model_freeze`**: Freezing strategy (e.g., `LoRA+r4+a1+d0.25+attn`).
- **`--gpu_id`**: GPU to use for training (default: `0`).
