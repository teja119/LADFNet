# LADFNet: Local Attention Dual Fusion Network for Medical Image Segmentation

LADFNet is an enhanced **Vision-Language Medical Image Segmentation** framework built upon **LViT (Language meets Vision Transformer)**. The model introduces efficient local attention, dual-decoder feature fusion, and an adaptive loss function to improve lesion segmentation performance on medical imaging datasets.

> **Status:** Research Project  
> **Framework:** PyTorch  
> **Task:** Medical Image Segmentation

---

## Overview

Medical image segmentation is a challenging task due to limited annotated datasets, blurred lesion boundaries, and high computational requirements of transformer-based architectures.

LADFNet addresses these limitations by introducing:

- **Window-based Local Self-Attention (WLSA)** for efficient local feature modeling
- **Dual Decoder Architecture** for complementary boundary and structural learning
- **Adaptive Boundary-Aware Focal Tversky (ABAFT) Loss** for improved optimization
- **Vision-Language Fusion** using BERT text embeddings and image features

---

## Key Contributions

### Window-based Local Self-Attention (WLSA)

The original LViT uses a Pixel-Level Attention Module (PLAM). LADFNet replaces it with WLSA, allowing each feature to attend only to a local window instead of the entire feature map.

**Benefits**

- Lower computational complexity
- Better spatial locality
- Reduced overfitting
- Improved boundary preservation

---

### Dual Decoder Architecture

Instead of a single decoder, LADFNet employs two parallel decoders:

- **Detail Decoder**
  - Focuses on lesion boundaries
  - Uses WLSA

- **Structure Decoder**
  - Captures global lesion structure
  - Uses lightweight convolution blocks

Both outputs are fused using a learned spatial fusion module.

---

### Adaptive Boundary-Aware Focal Tversky (ABAFT) Loss

The proposed loss combines:

- Tversky Loss
- Focal Tversky Loss
- Boundary Loss

Adaptive learnable weights automatically balance these objectives during training.

---

## Architecture

```
                Clinical Text
                      │
                  BERT Encoder
                      │
                      ▼
Image ──► CNN + Vision Transformer Encoder
                      │
             Cross-Modal Fusion
                      │
          Window Local Attention
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
 Detail Decoder              Structure Decoder
        │                           │
        └─────────────┬─────────────┘
                      ▼
              Learned Fusion Module
                      │
                Segmentation Mask
```

---

## Experimental Results

Dataset:

- MosMedData+

Performance:

| Model | Dice Score |
|--------|-----------:|
| LViT | 74.57% |
| **LADFNet** | **74.92%** |

LADFNet improves the Dice score while maintaining computational efficiency through localized attention.

---

## Technologies Used

- Python
- PyTorch
- Vision Transformer (ViT)
- BERT
- Albumentations
- OpenCV
- NumPy
- CUDA

---

## Repository Structure

```
LADFNet/
│
├── dataset/
├── models/
├── networks/
├── losses/
├── utils/
├── train.py
├── test.py
├── inference.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/teja119/LADFNet.git

cd LADFNet
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

---

## Evaluation

```bash
python test.py
```

---

## Dataset

Experiments were conducted on the **MosMedData+** medical image segmentation dataset.

The dataset contains CT scans with accompanying clinical text annotations for vision-language segmentation.

---

## Future Work

- Extend evaluation to QaTa-COV19 and ESO-CT datasets
- Support 3D CT/MRI segmentation
- Improve multimodal text-image fusion
- Integrate self-supervised pretraining

---

## Citation

If you use this work, please cite:

```
@article{ladfnet2025,
  title={LADFNet: Local Attention Dual Fusion Network for Vision-Language Medical Image Segmentation},
  author={Tejas Mahajan et al.},
  year={2025}
}
```

---

## Acknowledgements

This work builds upon the **LViT (Language meets Vision Transformer)** framework and extends it with efficient local attention, dual-decoder fusion, and adaptive optimization strategies for medical image segmentation.
