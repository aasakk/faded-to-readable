# Manuscript Segmentation and Section Classification System

### Faded to Readable – AI-Based Document Enhancement Pipeline

BCD-U-Net + Transformer Integration

---

## Overview

This project presents an end-to-end AI pipeline for enhancing and structurally organizing scanned academic manuscripts.

The system integrates document image segmentation and semantic paragraph classification into a unified workflow that transforms noisy manuscript scans into structured, readable, and annotated outputs.

### The system performs:

1. Document segmentation using BCD-U-Net to clean and isolate text regions from noisy scans
2. OCR extraction using Tesseract to convert segmented text into digital form
3. Paragraph-level classification using a Transformer model
4. Visual annotation of manuscript pages with color-coded section labels

Users can upload scanned manuscript pages through a Flask-based web interface and receive both structured text output and visually annotated images.

---

## System Architecture

Image → BCD-U-Net Segmentation → OCR (Tesseract) → Paragraph Splitting → Transformer Classification → Visual Annotation → Web Display

---

## Project Structure

```
faded_to_readable/
│
├── frontend1.py
├── label_mapping.json
├── tokenizer_config.json
├── vocab.txt
│
├── tf_model/
│   ├── config.json
│
├── static/
│   ├── uploads/
│   └── results/
│
├── templates/
│   └── index1.html
│
├── training_scripts/   (optional – model training files)
│
├── requirements.txt
└── README.md
```

Note: Pretrained model weights are not included in this repository due to size limitations. See Model Weights section below.

---

## Model Weights

The following pretrained weights are required to run the system:

* weight_text.hdf5 (BCD-U-Net segmentation model)
* manuscript_classifier.h5 (Transformer classifier)
* tf_model/tf_model.h5

Download the pretrained weights from:
(https://drive.google.com/drive/folders/1KpAgThmyI9tf0My4Jq1P_vu-OmSpew6a?usp=sharing)

After downloading, place them in the root project directory as shown in the structure above.

---

## Technologies Used

| Component           | Technology                   |
| ------------------- | ---------------------------- |
| Web Framework       | Flask                        |
| Segmentation Model  | BCD-U-Net (TensorFlow/Keras) |
| Text Classification | Transformer (TensorFlow)     |
| OCR                 | Tesseract                    |
| Image Processing    | OpenCV, Pillow               |
| NLP                 | Transformers                 |
| ML Utilities        | NumPy, Scikit-learn          |

---

## Requirements

Install dependencies:

```
pip install -r requirements.txt
```

Or manually:

```
pip install flask tensorflow keras numpy pillow pytesseract opencv-python transformers torch torchvision scikit-learn
```

---

## Tesseract OCR Setup

### Windows

Download:
[https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

Add this inside frontend1.py:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### Linux

```
sudo apt install tesseract-ocr
```

---

## How to Run

1. Navigate to the project directory:

```
cd faded_to_readable
```

2. Start the Flask server:

```
python frontend1.py
```

3. Open browser:

```
http://127.0.0.1:5000/
```

4. Upload a manuscript image (.jpg, .png, .jpeg)

5. View:

   * Classified paragraphs
   * Annotated manuscript image

---

## Processing Workflow

### Step 1 – Image Upload

Image is stored temporarily in `/static/uploads/`.

### Step 2 – Segmentation

BCD-U-Net removes background noise and enhances text regions.
Model trained on the DIBCO dataset for document binarization.

### Step 3 – OCR

Tesseract extracts text and provides bounding box coordinates.

### Step 4 – Paragraph Classification

Transformer model predicts:

* Title
* Introduction
* Objective
* Methodology
* Results
* Discussion
* References

Label mapping handled via `label_mapping.json`.

### Step 5 – Annotation

Bounding boxes are drawn around detected paragraphs with section-specific colors.
Output saved in `/static/results/`.

---

## Example Execution Log

```
[INFO] Upload received: manuscript_page.png
[INFO] Performing segmentation...
[INFO] OCR extraction completed.
[INFO] 6 paragraphs detected.
[INFO] Predicted labels: ['Title', 'Introduction', 'Objective', 'Methodology', 'Results', 'References']
[INFO] Output saved to /static/results/result_manuscript_page.png
```

---

## Best Practices for Better Results

* Use scans with at least 300 DPI resolution
* Avoid handwritten notes and heavy annotations
* Process one page at a time
* Fine-tune classifier for domain-specific datasets

---

## Future Enhancements

* Multi-page PDF support
* Layout detection for multi-column manuscripts
* User-feedback-driven adaptive retraining
* Accuracy evaluation metrics
* Cloud deployment support

---

## Authors

Aasavari Khire – 23BCB0105
Drashi Manoria – 23BCB0146
Yash Khose – 23BCE0625

---



