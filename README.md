# DualCaMIL
DualCaMIL: Dual-Level Causal Multi-Instance Learning for Patient-Level Diagnosis in Reflectance Confocal Microscopy

## Repository Structure

```text
train_causal.py          
test_causal.py          
requirements.txt         
models/
  causal_net.py          
  layers.py              
dataset/
  causal_data.py        
```

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

The code was tested with Python 3.12, PyTorch 2.5.1, torchvision 0.20.1, and transformers 4.51.3.

## Data Format

Data files are not included in this repository.

By default, the training script expects the following layout:

```text
Datasets/
  RCM/
    DataFile.csv
    Patient-001/
      image_001.png
      image_002.png
  Path/
    DataFile.csv
    tiles/
      class_or_case_folder/
        tile_001.jpg
```

Each CSV file should contain at least:

```text
ImagePath,Label
Datasets/RCM/Patient-001/image_001.png,0
Datasets/RCM/Patient-001/image_002.png,0
```

For RCM data, the patient ID is inferred from the parent directory of each image path.

## Training

Run training with the default paths:

```bash
python train_causal.py
```

Or specify custom data paths:

```bash
python train_causal.py \
  --rcm-csv Datasets/RCM/DataFile.csv \
  --rcm-root Datasets/RCM \
  --path-csv Datasets/Path/DataFile.csv \
  --path-root Datasets/Path/tiles \
  --output-dir outputs/causal_mil
```

The best model checkpoint is saved to:

```text
outputs/causal_mil/best/model.pt
```

## Evaluation

Evaluate a trained checkpoint:

```bash
python test_causal.py \
  --ckpt outputs/causal_mil/best/model.pt \
  --split-file outputs/convnextv2_mil_patient/splits.json
```

The evaluation script reports AUC, accuracy, macro F1, and saves patient-level probabilities as JSON.

## Notes

- The default backbone is `facebook/convnextv2-tiny-1k-224`.
- The backbone weights are downloaded by `transformers` on first use.
- Training works best with a CUDA GPU.
- Data, checkpoints, and generated outputs are intentionally ignored by Git.

## License

Add a license file before publishing. MIT License is a common choice for research code when you want others to freely use, modify, and redistribute the code with attribution.
