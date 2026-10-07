# CL-MobileViT: Wheat Leaf Disease Recognition and Safety-Gated Pesticide Retrieval

This project uses MobileViT as its visual backbone for six-class wheat leaf image classification. Training combines cross-entropy with supervised contrastive learning. During inference, the system produces a Grad-CAM visualization and then passes the classification result to a safety-gated retrieval layer backed by a pesticide-registration CSV file.

> This repository snapshot includes model weights, but it **does not include the training dataset or `data/knowledge/China_pesticides_wheat.csv`**. Classification and Grad-CAM can therefore run once an input image is provided, while pesticide retrieval fails closed and returns no candidate records when the CSV is unavailable. Training, standalone knowledge-layer queries, and knowledge-layer unit tests require the corresponding data to be added first.

## System Workflow

```text
Wheat leaf image
    ↓
CL-MobileViT six-class classification
    ↓
Grad-CAM attention-region visualization
    ↓
Confidence, healthy-class, and disease-name safety gates
    ↓
Exact retrieval from the wheat pesticide-registration CSV
    ↓
Candidate registration records and safety notices (not an automatic prescription)
```

The CSV knowledge layer is placed after the neural network. It uses only the Python standard library to read the data, does not participate in the model forward pass, and adds no model parameters. The current implementation is a lightweight and traceable rule-based retrieval prototype; it should not be described as an implemented graph-database knowledge graph.

## Implementation Overview

- **Visual backbone:** MobileViT with `xx_small`, `x_small`, and `small` configurations. ECA channel attention is added to the inverted residual blocks.
- **Training objective:** Two augmented views are generated for each training image. The model outputs both classification logits and a 128-dimensional projection feature. The total loss is `cross-entropy (label smoothing=0.1) + sup_con_weight × supervised contrastive loss`.
- **Validation:** Validation uses a single image view. Loss, Accuracy, macro-averaged Precision, Recall, and F1 are printed, but `best_model.pth` is currently selected using validation Accuracy only.
- **Explanation and retrieval:** Grad-CAM hooks are registered on `model.layer_5`. The classification result is then passed to a safety-medication rule layer that is separate from the model.
- **Fail-closed behavior:** If the CSV is missing, malformed, or unreadable, classification and Grad-CAM can still be produced, but the candidate list remains empty and the report status is `knowledge_base_unavailable_manual_review`.

## Project Structure

| Path | Purpose |
| --- | --- |
| `model.py` | MobileViT, ECA, classification head, and contrastive-learning projection head |
| `model_config.py` | Network configurations for `xx_small`, `x_small`, and `small` |
| `transformer.py` | Transformer encoder implementation |
| `my_dataset.py` | Two-view training data and single-view validation data |
| `utils.py` | Data splitting, supervised contrastive loss, training, and validation |
| `train.py` | Training entry point; currently fixed to `mobile_vit_small` |
| `predict.py` | Classification, Grad-CAM, safety-gated retrieval, and report export |
| `medication_knowledge.py` | Disease mapping, CSV validation, candidate filtering, ranking, and report generation |
| `test_medication_knowledge.py` | Knowledge-layer unit tests that depend on the real CSV file |
| `class_indices.json` | Class-index mapping for the current six-class model |
| `weights/` | Pretrained weights and existing six-class checkpoints |
| `outputs/smoke_test/` | Historical smoke-test outputs provided only as format examples |

The current class order is:

```text
0 Brown Rust
1 Healthy
2 Leaf Blight
3 Mildew
4 Septoria
5 Yellow Rust
```

## Environment Setup

Run the commands from the project root. The Python dependencies below are derived from the actual imports in the code:

- PyTorch and Torchvision
- Pillow and NumPy
- tqdm and Matplotlib
- TensorBoard for training logs

First install versions of `torch` and `torchvision` that match your CPU/CUDA environment using the official PyTorch installation instructions. Then install the remaining dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pillow numpy tqdm matplotlib tensorboard
```

Run a quick compilation check:

```powershell
python -m compileall -q .
```

## Data Preparation

### Training Images

By default, `train.py` reads `data/Wheat`. Each first-level subdirectory represents one class. Directory names are sorted lexicographically to generate labels, and the project-root `class_indices.json` is overwritten:

```text
data/
└── Wheat/
    ├── Brown Rust/
    │   ├── image_001.jpg
    │   └── ...
    ├── Healthy/
    ├── Leaf Blight/
    ├── Mildew/
    ├── Septoria/
    └── Yellow Rust/
```

Data-loading rules:

- Supported extensions are `.jpg`, `.JPG`, `.png`, and `.PNG`.
- Every image must be in RGB mode. Grayscale or other image modes raise an error.
- For each class, 20% of the images are sampled for validation using the fixed random seed `0`; the remaining images are used for training.
- To remain compatible with the existing six-class weights, directory names and their sorted order should match the class list above.

### Pesticide-Registration CSV

The default path is:

```text
data/knowledge/China_pesticides_wheat.csv
```

The CSV must use UTF-8 or UTF-8 with BOM and contain the following headers:

```text
登记证号, 农药名称, 农药类别, 登记证持有人, 剂型, 毒性,
总有效成分含量, 有效成分, 有效成分英文名, 作物/场所, 防治对象,
用药量（制剂量/亩）, 施用方法, 批准日期, 最新批准日期,
使用技术要求, 注意事项, 中毒急救措施, 储存和运输方法, 备注, 使用方法序号
```

If the CSV is stored elsewhere, specify it with `predict.py --pesticide-csv` or `medication_knowledge.py --csv`.

## Training

The current training entry point is fixed to `mobile_vit_small`. By default, it initializes the model from the 1,000-class `weights/mobile_vit_small.pt` checkpoint and removes the incompatible classification-head and projection-head parameters:

```powershell
python train.py `
  --data-path "data\Wheat" `
  --num_classes 6 `
  --epochs 50 `
  --batch-size 16 `
  --lr 0.0002 `
  --sup_con_weight 0.5 `
  --weights "weights\mobile_vit_small.pt" `
  --device cuda:0
```

Training produces:

- `weights/best_model.pth`: the checkpoint with the highest validation Accuracy.
- `weights/latest_model.pth`: the checkpoint from the most recent epoch.
- `runs/`: TensorBoard logs.
- `class_indices.json`: a class mapping regenerated from the dataset directory names.

View the logs with:

```powershell
tensorboard --logdir runs
```

Training notes:

- The bundled `best_model.pth` and `latest_model.pth` files are six-class `xx_small` weights with a classifier shape of `6 × 320`, whereas `train.py` is fixed to train `small`. Running training directly will overwrite those filenames with new `small` checkpoints. Back up the existing weights or change the output filenames first.
- `--freeze-layers` currently uses `argparse` with `type=bool`. To leave the backbone unfrozen, omit the option entirely. Do not pass `--freeze-layers False`, because a non-empty string may still be interpreted as `True`. Use `--freeze-layers True` to freeze the backbone.
- Start training from the project root. Otherwise, `class_indices.json`, `weights/`, and `runs/` are written relative to the current working directory.

## Inference

Minimal command:

```powershell
python predict.py --image "D:\wheat_images\sample.jpg"
```

Default behavior:

- Uses `weights/best_model.pth`.
- Automatically identifies `xx_small`, `x_small`, or `small` from the classifier input dimension. The current `best_model.pth` is detected as `xx_small`.
- Uses an image size of `224 × 224` and automatically selects CUDA or CPU.
- Uses a safety confidence threshold of `0.80` and returns at most three candidate records.
- Writes outputs to `outputs/`.

Full example:

```powershell
python predict.py `
  --image "D:\wheat_images\sample.jpg" `
  --weights "weights\best_model.pth" `
  --model-variant auto `
  --class-indices "class_indices.json" `
  --pesticide-csv "data\knowledge\China_pesticides_wheat.csv" `
  --confidence-threshold 0.85 `
  --top-k 5 `
  --device cpu `
  --output-dir "outputs\sample"
```

Each successful model run generates:

- `grad_cam_result.png`: the original image, Grad-CAM heatmap, and overlay.
- `medication_recommendation.json`: model information, per-class probabilities, knowledge-layer status, and candidate records.
- `medication_recommendation.txt`: a human-readable text report.


## Standalone Knowledge-Layer Queries and Tests

Once the CSV is available, the rule layer can be queried without loading the neural network:

```powershell
python medication_knowledge.py --disease Mildew --confidence 0.95
```

Specify an external CSV with:

```powershell
python medication_knowledge.py `
  --disease "Yellow Rust" `
  --confidence 0.95 `
  --csv "D:\data\China_pesticides_wheat.csv" `
  --output-dir "outputs\knowledge_test"
```

Run the knowledge-layer unit tests with:

```powershell
python -m unittest -v test_medication_knowledge.py
```

Because the default CSV is missing from this repository snapshot, the tests currently raise `FileNotFoundError` during `setUpClass` and do not execute the seven test cases. After the CSV is added, the tests cover the healthy class, low confidence, missing exact mappings, powdery mildew, stripe rust, and leaf rust rules.

## Usage Boundaries

- Grad-CAM is a diagnostic visualization of model attention. On its own, it does not demonstrate reliable lesion localization or provide a causal explanation.
- The output contains candidate registration records, not an automatic prescription. The CSV does not include registration expiry dates or live registration status, so the system cannot claim that a record remains valid on the day of use.
- Before any real-world application, verify the registration number, current official label, crop, control target, dosage, application method, pre-harvest interval, and local regulations, and consult a qualified crop-protection professional.
- Candidate ranking considers exact matching, a more conservative toxicity order, newer approval dates, and registration numbers. It is used only for display and does not represent efficacy ranking or product endorsement.

## Current Verification Status

The following facts have been confirmed for the current code and local environment:

- `python -m compileall -q .` passes.
- `predict.py --help` parses all inference arguments successfully.
- The existing `best_model.pth` and `latest_model.pth` classifiers have shape `6 × 320`, corresponding to `xx_small`.
- Knowledge-layer tests stop because the default CSV is missing; this is not reported as a successful knowledge-retrieval test run.
- The model has not been retrained during this review, and no new Accuracy, Macro-F1, or other real performance results have been produced.
