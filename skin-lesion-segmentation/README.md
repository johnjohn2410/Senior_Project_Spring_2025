🩺 Skin Lesion Segmentation using MONAI
📌 Project Overview
This project focuses on automating skin lesion segmentation from dermoscopic images using deep learning. We utilize the ISIC Archive dataset and build a U-Net-based segmentation model with MONAI (Medical Open Network for AI) and PyTorch. The model is optimized for medical imaging applications.

🎯 Goals
✅ Develop a U-Net segmentation model for skin lesion detection.
✅ Utilize MONAI's specialized tools for medical AI.
✅ Apply data augmentation to improve generalization.
✅ Train the model using GPU acceleration for efficiency.
✅ Deploy a real-time segmentation demo for predictions.

🚀 Installation & Setup
1️⃣ Clone the Repository
bash
git clone https://github.com/YOUR_GITHUB_USERNAME/skin-lesion-segmentation.git
cd skin-lesion-segmentation

2️⃣ Create a Virtual Environment & Install Dependencies
bash
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
pip install -r requirements.txt

3️⃣ Download the ISIC Dataset
Visit the ISIC Archive.
Download the ISIC dataset (images & segmentation masks).
Organize them as follows:
bash
├── data/
│   ├── images/   # ISIC training images
│   ├── masks/    # Corresponding segmentation masks
🧠 Model Architecture
The segmentation model is built using a U-Net architecture with EfficientNet as an encoder, designed to handle medical image segmentation effectively.

🛠️ Model Details
Encoder: EfficientNet-B4 (Pretrained on ImageNet)
Decoder: U-Net with transposed convolutions
Loss Function: Dice Loss + Binary Cross-Entropy (BCE)
Optimizer: AdamW with Cosine Annealing LR
Augmentations: Albumentations for medical image processing
📊 Training the Model
Run the training script:

bash
python src/train.py
This will:
✅ Load the ISIC dataset.
✅ Apply data augmentation.
✅ Train the MONAI U-Net model.
✅ Save the best model weights in /models/.

🎯 Making Predictions
To run inference on a new image:

bash
python src/predict.py --image path/to/image.jpg
This will output:
✅ The segmentation mask overlaid on the original image.
✅ Prediction confidence scores.

🔥 Performance Metrics
The model is evaluated using:
✔ Dice Coefficient (Higher is better)
✔ Jaccard Index (IoU)
✔ Sensitivity & Specificity

🚀 Goal: Achieve 90%+ Dice Score on validation data!

👨‍💻 Team Members
Name	Role
[John Ross]	Model Development	
Daniel Gutierrez Data Engineering & Preprocessing
Esteban Kott Training & Optimization
Joe Reyna Deployment & Web App
👥 Contributing
Want to contribute? Follow these steps:

Fork this repository.
Create a new branch:
bash
git checkout -b feature-branch-name
Make changes & commit:

bash
git add .
git commit -m "Added new feature"
Push to GitHub:

bash
git push origin feature-branch-name
Submit a Pull Request (PR) for review.
⚖️ License
📜 This project is licensed under the MIT License – free to use and modify.

🛠️ Resources
📂 ISIC Dataset → ISIC Archive
📖 MONAI Documentation → MONAI.io
🔥 PyTorch Official Docs → PyTorch.org