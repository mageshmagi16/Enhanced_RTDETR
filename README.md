# Enhanced RT-DETR: Real-Time Object Detection Transformer
 
**Real-Time Object Detection using Transformer-based Architecture**

This project implements **RT-DETR**, an end-to-end **NMS-free object detection model** using PyTorch. It achieves better accuracy and real-time inference speed. The repository includes the improved base model, training scripts, inference tools, and a detailed blog post explaining the model and improvements.

## Learn More  
 
For a detailed explanation of the **RT-DETR architecture, training process, and improvements**, check out my Medium blog:  

[Enhanced RT-DETR: Real-Time Object Detection Transformer (Medium Blog)](https://medium.com/@maheshg16/enhanced-rt-detr-real-time-object-detection-df54ec83c2b9)

  
---

## Installation & Setup

### 1. Clone the Repository
```
git clone https://github.com/YourUsername/RT-DETR.git
cd RT-DETR
```

### 2. Create a Virtual Environment (Recommended)
```
python -m venv venv
source venv/bin/activate      # On Mac/Linux
venv\Scripts\activate         # On Windows
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```
### 3. Model Weights
The trained RT-DETR weights are stored in this repository using Git LFS (Large File Storage).
Make sure you have Git LFS installed before cloning:

```
# Install Git LFS (if not already installed)
git lfs install

# Pull LFS files (model weights, checkpoints, etc.)
git lfs pull

```

### Project Files
configs/ - YAML configuration files for training and inference.

tools/ - Scripts for training, testing, and inference.

images/ - Sample images for inference and blog illustrations.

videos/ - Sample videos for inference.

checkpoints/ - Trained model weights and checkpoints.

rtdetr_blog/ - Markdown blog post explaining RT-DETR architecture and improvements.

