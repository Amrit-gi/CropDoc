# CropDoc: Academic Review Panel Presentation Guide

A step-by-step presentation script, slide outline, and live demo guide designed for academic review panels, faculty evaluations, and capstone presentations.

---

## 📋 Presentation Overview
- **Project Title**: CropDoc — Betel Leaf Disease Detection System with Severity Scoring, Explainability, and Treatment Advice
- **Target Duration**: 10 – 12 Minutes (8 min slides + 3 min live demo + Q&A)
- **Presenter**: Project Author / Student Team

---

## 🖥️ Slide-by-Slide Presentation Outline

### Slide 1: Title Slide
- **Project Name**: CropDoc
- **Subtitle**: On-Device Pathology Diagnosis, Severity Estimation, Explainability & Treatment Advice for Betel Leaf (*Piper betle*)
- **Presenter Name & Department**
- **Date & Academic Review Session**

---

### Slide 2: Real-World Problem Statement
- **Economic Value**: Betel leaf (*Piper betle*) is a high-value cash crop deeply rooted in South Asian agriculture and exports.
- **The Challenge**: Betel vine conservatories (*bareja*) suffer heavily from **Leaf Rot** (*Phytophthora parasitica*) and **Leaf Spot** (*Xanthomonas campestris* / *Colletotrichum*).
- **The Ground Reality**: Farmers in remote rural areas lack timely pathological expertise. Misdiagnosis leads to indiscriminate pesticide sprays, financial loss, and soil damage.

---

### Slide 3: Research Gap & Novelty (Key Academic Slide)
- **Literature Review Synthesis** (Survey of 10 Papers):
  - *Classification-only papers* (rows 1-5, 7, 8) never add severity scoring.
  - *Severity-focused papers* (rows 9-10) use heavy segmentation models and cover only one disease.
  - *Mobile app papers* (rows 6, 8) name the disease only — no severity level, explainability, or treatment dosage.
- **Our Novelty Statement**:
  > *"No existing paper combines disease classification, severity scoring, explainability, and treatment recommendation in a single lightweight, deployable pipeline — this is the specific research gap CropDoc addresses."*

---

### Slide 4: System Architecture & Pipeline
Show the system workflow diagram:

```
[Input Leaf Photo] 
       │
       ▼
1. Validation Stage ──────► (Rejects corrupted/blank images)
       │
       ▼
2. MobileNetV2 Classifier ──► Disease Prediction (Healthy / Rot / Spot) + Confidence %
       │
       ▼
3. HSV Severity Engine ───► Lesion Area % (Healthy, Mild, Moderate, Severe)
       │
       ▼
4. Grad-CAM Module ───────► Heatmap Visual Overlay
       │
       ▼
5. Treatment Lookup ──────► Actionable Chemical, Organic & Cultural Advice
       │
       ▼
[Streamlit Web App / Farmer Interface]
```

---

### Slide 5: Methodology & Model Design
1. **MobileNetV2 Classifier**:
   - Transfer learning using ImageNet pretrained weights.
   - **~2.26 Million Parameters** (lightweight for CPU/mobile deployment).
   - GlobalAveragePooling2D + Dropout (0.3) + Dense Classifier.
2. **Color-Thresholding Severity Estimator**:
   - Converts RGB to **HSV Color Space**.
   - Segments healthy green tissue vs. brown/dark necrotic spots.
   - Calculates exact surface area percentage:
     $$\text{Severity Ratio (\%)} = \frac{\text{Lesion Surface Area Count}}{\text{Total Leaf Surface Area Count}} \times 100$$
3. **Grad-CAM Explainability**:
   - Extracts activation gradients from the last convolutional layer (`Conv_1`).
   - Overlays a JET heatmap highlighting regions of interest (spots/rot).

---

### Slide 6: Experimental Setup & Dataset
- **Dataset Source**: Mendeley Data (*vpzkntzjty*) — 12,222 images total.
- **Experimental Dataset Split**: 2,037 unaugmented field images across 3 classes:
  - `Healthy_Leaf`: 1,080 images
  - `Leaf_Spot`: 688 images
  - `Leaf_Rot`: 269 images
- **Train / Validation / Test Split**: 80% Train / 10% Validation / 10% Test.

---

### Slide 7: Empirical Results & Metric Charts
- **Test Accuracy**: **82.32%**
- **Test Precision**: **85.63%**
- **Test Recall**: **79.01%**
- **Inference Speed**: **< 50 milliseconds per image on standard CPU**.

*(Embed `results/confusion_matrix.png` and `results/accuracy_loss_curve.png` on this slide)*

---

### Slide 8: Live Demonstration (3 Minutes)
1. Switch to browser running `streamlit run app/streamlit_app.py`.
2. Upload a sample leaf photo from `data/sample_images/`.
3. Show the panel:
   - Instant Disease Diagnosis & Confidence Score.
   - Severity Score & Color-coded Severity Badge.
   - Side-by-side **Grad-CAM Heatmap Overlay**.
   - Actionable 4-part **Treatment Card** (Chemical, Organic, Cultural, Preventive).

---

### Slide 9: Code Structure & Engineering Quality
- **Modular Python Architecture**: Clean separation of `src/` modules.
- **Centralized Configuration**: `config.py` defines all hyperparameters and paths.
- **Decoupled Data Rules**: Treatment recommendations stored in JSON (`treatment_rules.json`).
- **Automated Testing**: 100% pass rate on Pytest suite (`pytest tests/test_pipeline.py`).

---

### Slide 10: Conclusion & Future Scope
- **Conclusion**: Successfully built and validated a 4-in-1 lightweight pathology diagnostic pipeline for betel leaf farming.
- **Future Directions**:
  1. Export MobileNetV2 to **TensorFlow Lite (.tflite)** for offline Android/iOS app integration.
  2. Integrate multi-spectral drone camera inputs for large-scale conservatory monitoring.

---

## ❓ Common Faculty Review Q&A Guide

### Q1: Why did you choose MobileNetV2 instead of heavier models like ResNet50 or VGG16?
> **Answer**: MobileNetV2 has ~2.26 million parameters compared to 25 million for ResNet50 and 138 million for VGG16. For real-world deployment in rural farms, the model must run fast on standard mobile CPUs without requiring expensive cloud GPUs or high battery draw. MobileNetV2 uses depthwise separable convolutions, reducing compute complexity by ~8x while maintaining competitive accuracy (82.3% test accuracy).

### Q2: Why did you use HSV color-thresholding for severity instead of U-Net or DeepLabv3 segmentation?
> **Answer**: Neural segmentation models like U-Net require pixel-by-pixel mask annotations for thousands of images, add tens of millions of parameters, and slow down inference. HSV color thresholding requires **zero extra model parameters**, processes in under 10 milliseconds, and is fully explainable to agricultural experts by measuring actual green surface vs. diseased lesion surface area ratios.

### Q3: Why is Leaf Rot recall lower than Healthy Leaf?
> **Answer**: The raw field dataset has a natural class imbalance (1,080 Healthy vs 269 Leaf Rot). While data augmentation helps, Healthy Leaf and Leaf Spot have larger sample coverage. In future work, synthetic minority oversampling (SMOTE) or targeted field data collection for Leaf Rot will improve recall balance.

### Q4: How does Grad-CAM benefit the farmer or extension officer?
> **Answer**: Black-box AI models often suffer from distrust. Grad-CAM computes activation gradients at the last conv layer (`Conv_1`) and projects a visual heatmap overlay directly onto the leaf. This proves to the farmer *why* the model predicted Leaf Spot by highlighting the exact diseased lesion spots.
