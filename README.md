# 🖼️ Image Super-Resolution

## 📌 Overview
The aim of this project is to replicate and improve the results and models used in the 📖 [paper (C. Ledig et al.)](https://arxiv.org/pdf/1609.04802). The goal is to upscale low-resolution images to high-resolution using a deep convolutional neural network (CNN) and generative adversarial networks (GANs).

## 📂 Setup
To run this project, ensure that the following folders exist in your Google Drive:

1. **dataset_super_resolution/**
   - Contains the **DIV2K** dataset with:
     - 800 training images (low-resolution & high-resolution pairs)
     - 100 validation images (low-resolution & high-resolution pairs)
   - [Download dataset](https://drive.google.com/drive/folders/1MdcQS_9Mt89KnwyUSKtPyU-KNVmmG-uZ?usp=sharing)

2. **Test/**
   - Contains 5 test images in 4 versions:
     - Low-resolution
     - Bicubic upsampled
     - CNN-generated super-resolution
     - Ground-truth high-resolution
   - [Download test set](https://drive.google.com/drive/folders/1XSEnp1otkVy6r3Gf4XvsYF-B_CluJZ-p?usp=sharing)

## ⚙️ Running the Notebook
To execute the **DL_Final_Version.ipynb** notebook:

1. Open Google Colab and mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Ensure dependencies are installed:
   ```python
   %pip install imageio pytorch-ignite
   ```

3. Load and verify dataset images:
   ```python
   im1_high_resolution = imread(dataset_path + "training_HR/0001.png")
   plt.imshow(im1_high_resolution)
   ```

## 🏗️ Model Training
The project includes three approaches:

1. **CNN Model (Supervised Learning)**
   - Uses **Mean Squared Error (MSE) loss** for training.
   - Produces outputs similar to bicubic interpolation.
   - The model is trained using **PyTorch** with data augmentation via random patch extraction.

2. **SRGAN (Generative Adversarial Network)**
   - Introduces an adversarial loss to improve fine textures.
   - Uses a **feature extractor (VGG19)** for perceptual loss.
   - Discriminator learns to distinguish real vs. generated high-resolution images.

3. **WGAN (Wasserstein GAN)**
   - Implements Wasserstein loss with gradient penalty.
   - Aims to stabilize GAN training.

## 📊 Results & Comparisons
We evaluate the models using:
- **PSNR (Peak Signal-to-Noise Ratio)** to measure image quality.
- Visual comparison of **low-res, bicubic, CNN, and GAN outputs**.

### Sample Output (Super-Resolution Images)
#### **Example from Test Set**
| Low-Resolution | Bicubic Upsampling | CNN Super-Resolution | Ground Truth |
|---------------|-------------------|----------------------|-------------|
| ![Low-Res](test_images/low_res.png) | ![Bicubic](test_images/bicubic.png) | ![CNN Output](test_images/cnn_output.png) | ![Ground Truth](test_images/ground_truth.png) |

## 📌 Conclusion
- The CNN model produces **decent upscaling**, but struggles with fine textures.
- SRGAN introduces **high-frequency details**, but requires careful tuning to avoid artifacts.
- WGAN shows promise but needs **further optimization**.

## 👥 Contributors
-  [Riccardo Riglietti]()
-  [Antonio Scardino](https://github.com/antoscardi)

---
🔧 **Developed in Google Colab with PyTorch**

