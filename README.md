# 🩺 Skin Lesion Segmentation

## 📌 Project Overview
This project aims to develop a **deep learning-based segmentation model** to accurately identify and delineate **skin lesions** from **dermoscopic images** using the **ISIC Archive dataset**. The model is built using **MONAI (Medical Open Network for AI)** and **PyTorch**, optimized for medical imaging.

### 🎯 **Goals:**
- Develop a **U-Net segmentation model** for **skin lesion detection**.
- Use **MONAI’s medical AI tools** to improve accuracy.
- Apply **data augmentation** to enhance generalization.
- Train the model on **GPU for fast performance**.
- Deploy a **demo application for real-time predictions**.

---



## 🚀 **Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/skin-lesion-segmentation.git
cd skin-lesion-segmentation
```
## 2️⃣ Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## 3️⃣ Download the ISIC Dataset
Visit ISIC Archive.
Download images and corresponding segmentation masks.
Organize them in the /data/images/ and /data/masks/ directories.
🧠 Model Architecture
The segmentation model is based on a U-Net architecture with EfficientNet as the encoder, optimized for medical imaging.

🛠️ Model Details
Encoder: EfficientNet-B4 (Pretrained on ImageNet)
Decoder: U-Net with transposed convolutions
Loss Function: Dice Loss + Binary Cross-Entropy (BCE)
Optimizer: AdamW with Cosine Annealing LR
Augmentations: Albumentations for medical image processing
📊 Training the Model
Run the training script:

```bash
python src/train.py
```
This will:
Load the ISIC dataset.
Apply data augmentation.
Train the MONAI U-Net model.
Save the best model weights in /models/.
🎯 Making Predictions
To run inference on a new image:
```bash
python src/predict.py --image path/to/image.jpg
```
This will output:
The segmentation mask overlaid on the original image.
Prediction confidence scores.
🔥 Performance Metrics
The model is evaluated using:

Dice Coefficient (Higher is better)
Jaccard Index (IoU)
Sensitivity & Specificity
🚀 Goal: Achieve 90%+ Dice Score on validation data!

## Contributors
- [John Ross]
- [Daniel Gutierrez]
- [Joe Reyna]
- [Esteban Kott]

---

## Contact
For questions or suggestions, please contact daniel.gutierreziii01@utrgv.edu, john.ross01@utrgv.edu, Esteban.kott01@utrgv.edu, joe.reyna02@utrgv.edu
Faculty Advisor Dr. Pengfei Gu. pengfei.gu01@utrgv.edu

⚖️ License
📜 This project is licensed under the MIT License – free to use and modify.

🛠️ Resources
ISIC Dataset → https://challenge.isic-archive.com/data/
MONAI Documentation → https://monai.io/
PyTorch Official → https://pytorch.org/

