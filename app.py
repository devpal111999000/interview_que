from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import os
import gdown

app = Flask(__name__)

# Constants
MODEL_DIR = "t5_question_gen_model"
DRIVE_URL = "https://drive.google.com/drive/folders/18InW9XucY1lvlNAo6qzfshWyNpClNXMr?usp=sharing" 

def download_and_extract_model():
    if not os.path.exists(MODEL_DIR):
        print("Downloading model from Google Drive...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        gdown.download(DRIVE_URL, output="model.zip", quiet=False)
        import zipfile
        with zipfile.ZipFile("model.zip", "r") as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        os.remove("model.zip")
        print("Model downloaded and extracted.")

# Load model/tokenizer
download_and_extract_model()
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_questions(description, num_questions=10):
    input_text = f"Given the following job description, generate a specific technical interview question: {description}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=128,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=num_questions,
        early_stopping=True
    )

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    if not data or "description" not in data:
        return jsonify({"error": "Missing 'description' in request body."}), 400
    try:
        questions = generate_questions(data["description"])
        return jsonify({"questions": questions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return "T5 Interview Question Generator is running!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
