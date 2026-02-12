# Manuscript Segmentation and Section Classification System

Faded to Readable AI Project

**(BCD-U-Net + Transformer Integration)**

## Overview

This project integrates document image segmentation and semantic text classification into a unified AI-driven workflow for making research manuscripts and academic documents easier to read.

The system performs the following sequence of tasks automatically:

1. Image segmentation using **BCD-U-Net** to isolate clean text regions from noisy manuscript scans.
2. Optical Character Recognition (**OCR**) using **Tesseract** to convert extracted text into digital form.
3. Paragraph-level classification using a **Transformer-based model** that labels each paragraph as *Introduction*, *Objective*, *Methodology*, *Results*, *Discussion*, or *References*.
4. Visual annotation of the original manuscript image with color-coded bounding boxes for each classified section.

The entire process runs through a **Flask-based web interface**, allowing users to upload scanned manuscript pages and view structured, labeled outputs both textually and visually.

Training scripts for both models (BCD-U-Net and Transformer classifier) are also included in the project folder but are not required for running the deployed application, since pretrained models are already provided.

---

## Project Directory

```
faded to readable/
│
├── frontend1.py                # Flask backend integrating both models
│
├── weight_text.hdf5            # Pretrained BCD-U-Net model weights for segmentation
├── manuscript_classifier.h5    # Transformer model weights for paragraph classification
├── label_mapping.json          # Label index-to-name mappings for classifier output
├── tokenizer_config.json       # Tokenizer configuration (for text preprocessing)
├── vocab.txt                   # Tokenizer vocabulary
│
├── tf_model/                   # Folder containing transformer model components
│   ├── config.json
│   ├── tf_model.h5
│   └── other TensorFlow weight files
│
├── static/
│   ├── uploads/                # Temporarily stores uploaded manuscript images
│   └── results/                # Saves processed, annotated output images
│
├── templates/
│   └── index1.html             # Flask frontend for upload + result display
│
└── README.md                   # Documentation (this file)
```

*(Note: Training code files for both models are present in the project folder but are not shown in this directory structure since they are not required for execution when pretrained weights are available.)*

---

## Requirements

### Python Packages

Install all dependencies with:

```bash
pip install flask tensorflow keras numpy pillow pytesseract opencv-python transformers torch torchvision scikit-learn
```

| Library             | Purpose                          |
| ------------------- | -------------------------------- |
| Flask               | Hosts the web application        |
| TensorFlow / Keras  | Loads and runs both models       |
| NumPy               | Numerical and matrix operations  |
| Pillow (PIL)        | Image loading and annotation     |
| pytesseract         | OCR text extraction              |
| OpenCV              | Image preprocessing utilities    |
| Transformers        | Tokenizer and model inference    |
| Torch / torchvision | Auxiliary model support          |
| Scikit-learn        | Data utilities and preprocessing |

---

### Tesseract OCR Setup

Install **Tesseract OCR** for text extraction.

**Windows:**

* Download: [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
* Default path:

  ```
  C:\Program Files\Tesseract-OCR\tesseract.exe
  ```
* Add this line in `frontend1.py`:

  ```python
  import pytesseract
  pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```

**Linux/macOS:**

```bash
sudo apt install tesseract-ocr
```

---

## System Workflow

### Step 1: Image Upload

A manuscript image is uploaded via the Flask interface (`index1.html`). It is temporarily stored under `/static/uploads/`.

### Step 2: Segmentation with BCD-U-Net

The **BCD-U-Net model (`weight_text.hdf5`)** performs document binarization, cleaning, and segmentation, removing background noise and preserving text regions.
This model was trained on the DIBCO dataset for document enhancement and binarization tasks.

### Step 3: OCR Conversion

Using **Tesseract**, the clean segmented text is converted to digital text, which is split into paragraphs for classification.

### Step 4: Paragraph Classification

The **Transformer classifier (`manuscript_classifier.h5`)** processes each paragraph and predicts section labels such as *Introduction*, *Objective*, *Methodology*, *Results*, *Discussion*, or *References*.
Label mappings from numeric indices to section names are defined in `label_mapping.json`.

### Step 5: Post-Processing and Annotation

Paragraphs are matched to approximate bounding boxes on the original image using OCR coordinates.
Each box is drawn in a different color and labeled with its predicted section type.
The annotated output image is saved in `/static/results/`.

### Step 6: Output Visualization

The final page shows:

* Classified text paragraphs with labels.
* Annotated manuscript image with colored bounding boxes.

---

## Running the Application

1. **Navigate to the project directory:**

   ```bash
   cd faded to readable
   ```

2. **Run the Flask app:**

   ```bash
   python frontend1.py
   ```

3. **Access it in your browser:**

   ```
   http://127.0.0.1:5000/
   ```

4. **Upload and analyze a document:**

   * Upload an image (`.jpg`, `.png`, `.jpeg`)
   * Choose between *Text Cleaning* or *Segmentation*
   * Wait for processing and view the results

---

## Components Summary

| Component                        | Description                                                    |
| -------------------------------- | -------------------------------------------------------------- |
| weight_text.hdf5                 | Pretrained BCD-U-Net weights for text segmentation             |
| manuscript_classifier.h5         | Transformer model for paragraph-level section classification   |
| label_mapping.json               | Defines numeric-to-label mapping for model outputs             |
| tokenizer_config.json, vocab.txt | Tokenizer setup for text preprocessing                         |
| frontend1.py                     | Flask script integrating segmentation, OCR, and classification |
| index1.html                      | Upload and result display frontend                             |
| static/uploads                   | Stores user-uploaded images temporarily                        |
| static/results                   | Saves annotated output images                                  |

---

## Example Output

Terminal log during execution:

```
[INFO] Upload received: manuscript_page.png
[INFO] Performing segmentation with BCD-U-Net...
[INFO] OCR extraction completed.
[INFO] 6 paragraphs detected.
[INFO] Predicted labels: ['Title', 'Introduction', 'Objective', 'Methodology', 'Results', 'References']
[INFO] Annotated output saved to /static/results/result_manuscript_page.png
```

Display on webpage:

```
Paragraph 1 → Title  
Paragraph 2 → Introduction  
Paragraph 3 → Objective  
Paragraph 4 → Methodology  
Paragraph 5 → Results  
Paragraph 6 → References  
```

---

## Accuracy Tips

* Use high-quality scans (≥300 DPI).
* Avoid handwritten or overlapping annotations.
* For multi-page manuscripts, process one page at a time.
* Adjust paragraph splitting (`split("\n\n")`) based on text spacing.
* Fine-tune the classifier model for improved domain-specific accuracy.

---

## Future Work

* Add multi-page PDF support.
* Introduce user feedback–based retraining for adaptive labeling.
* Improve layout recognition for two-column manuscripts.
* Compute accuracy metrics for validation.
* Deploy to cloud servers for online access.

---

## Authors

Aasavari Khire 23BCB0105
Drashi Manoria 23BCB0146
Yash Khose 23BCE0625

