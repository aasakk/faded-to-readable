from flask import Flask, render_template, request
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import json
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------- Parameters ---------------- #
patch_size = 128
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ---------------- Load BCD-Unet model ---------------- #
model_text_clean = load_model("weight_text.hdf5")

# ---------------- Load Manuscript Model ---------------- #
label_file = "label_mapping.json"
tokenizer_dir = "./"       # folder containing tokenizer_config.json and vocab.txt
tf_model_dir = "./tf_model"  # folder containing full TF model

with open(label_file, "r") as f:
    label_mappings = json.load(f)

num_labels = label_mappings["num_labels"]

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

model_manuscript = TFAutoModelForSequenceClassification.from_pretrained(
    tf_model_dir,
    num_labels=num_labels,
    id2label=label_mappings["id2label"],
    label2id=label_mappings["label2id"]
)
model_manuscript.load_weights("manuscript_classifier.h5")

# ---------------- Flask Setup ---------------- #
app = Flask(__name__)

# ---------------- Helper function for BCD-Unet ---------------- #
def predict_full_image(img_path, save_path, model):
    img = load_img(img_path, color_mode='grayscale')
    img_array = img_to_array(img)
    orig_h, orig_w = img_array.shape[:2]

    pad_h = (patch_size - orig_h % patch_size) % patch_size
    pad_w = (patch_size - orig_w % patch_size) % patch_size
    img_padded = np.pad(img_array, ((0, pad_h), (0, pad_w), (0,0)), mode='reflect')

    padded_h, padded_w = img_padded.shape[:2]
    pred_mask = np.zeros((padded_h, padded_w, 1), dtype=np.float32)

    for i in range(0, padded_h, patch_size):
        for j in range(0, padded_w, patch_size):
            patch = img_padded[i:i+patch_size, j:j+patch_size, :]
            patch_norm = patch / 255.0
            patch_norm = np.expand_dims(patch_norm, axis=0)
            pred = model.predict(patch_norm)[0]
            pred_mask[i:i+patch_size, j:j+patch_size, :] = pred

    pred_mask = pred_mask[:orig_h, :orig_w, :]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    Image.fromarray(pred_mask.squeeze()).save(save_path)
    return save_path

# ---------------- Helper function for Manuscript Segmentation (paragraph-level) ---------------- #
def preprocess_and_predict_sections(img_path, save_path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # OCR: extract full text
    raw_text = pytesseract.image_to_string(img)

    # Split text into paragraphs intelligently
    raw_lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    paragraphs = []
    temp_para = ""
    for line in raw_lines:
        temp_para += " " + line
        if line.endswith(('.', '?', ':')):  # paragraph boundary
            paragraphs.append(temp_para.strip())
            temp_para = ""
    if temp_para:
        paragraphs.append(temp_para.strip())

    if not paragraphs:
        img.save(save_path)
        return [("No text detected", "")]

    # ---------------- Post-processing: Adjust first paragraph if too short ---------------- #
    if len(paragraphs) > 0 and len(paragraphs[0]) < 40:
        # Treat first paragraph as title
        first_para_as_title = True
    else:
        first_para_as_title = False

    # Tokenize paragraphs
    encoded_inputs = tokenizer(paragraphs, padding=True, truncation=True, max_length=256, return_tensors="tf")
    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]

    # Predict section labels
    preds = model_manuscript(input_ids=input_ids, attention_mask=attention_mask).logits
    pred_indices = tf.argmax(preds, axis=1).numpy()
    section_labels = [label_mappings["id2label"][str(idx)] for idx in pred_indices]

    # Force first paragraph as title if short
    if first_para_as_title:
        section_labels[0] = "title"

    # Force last paragraph as references if it starts like "References" or "Bibliography"
    if len(paragraphs) > 1:
        last_para = paragraphs[-1].strip().lower()
        if last_para.startswith("references") or last_para.startswith("bibliography"):
            section_labels[-1] = "references"

    # Optional: Annotate image per paragraph using rough bounding boxes
    boxes = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    para_boxes = []
    current_para = []
    prev_bottom = 0
    for i, text in enumerate(boxes['text']):
        if text.strip():
            top = boxes['top'][i]
            bottom = boxes['top'][i] + boxes['height'][i]
            if top - prev_bottom > 15 and current_para:
                para_boxes.append(current_para)
                current_para = []
            current_para.append((boxes['left'][i], top, boxes['left'][i]+boxes['width'][i], bottom))
            prev_bottom = bottom
    if current_para:
        para_boxes.append(current_para)

    # Draw rectangles per paragraph
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]
    for idx, para in enumerate(para_boxes):
        x0 = min(b[0] for b in para)
        y0 = min(b[1] for b in para)
        x1 = max(b[2] for b in para)
        y1 = max(b[3] for b in para)
        draw.rectangle([x0, y0, x1, y1], outline=colors[idx % len(colors)], width=2)
        if idx < len(section_labels):
            draw.text((x0, y0-10), section_labels[idx], fill=colors[idx % len(colors)], font=font)

    img.save(save_path)
    return list(zip(section_labels, paragraphs))



# ---------------- Routes ---------------- #
@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_path = None
    result_path = None
    manuscript_result = None

    if request.method == 'POST':
        file = request.files.get('file')
        model_choice = request.form.get('model_choice')

        if file and file.filename != '' and model_choice:
            uploaded_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(uploaded_path)
            uploaded_path = uploaded_path.replace("\\", "/")
            result_path = None

            if model_choice == "text_cleaning":
                result_path = os.path.join(RESULT_FOLDER, f"result_{file.filename}")
                predict_full_image(uploaded_path, result_path, model_text_clean)
                result_path = result_path.replace("\\", "/")

            elif model_choice == "segmentation":
                result_path = os.path.join(RESULT_FOLDER, f"result_{file.filename}")
                manuscript_result = preprocess_and_predict_sections(uploaded_path, result_path)
                result_path = result_path.replace("\\", "/")

    return render_template('index1.html',
                           uploaded=uploaded_path,
                           result=result_path,
                           manuscript_result=manuscript_result)

if __name__ == '__main__':
    app.run(debug=True)
