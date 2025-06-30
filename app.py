from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS
import torch

app = Flask(__name__)
CORS(app)

# Set device
device = 0 if torch.cuda.is_available() else -1
print("Device set to:", "CUDA" if device == 0 else "CPU")

# Use a smaller, faster model for summarization
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Backend is working! Use POST /generate to get summaries."})

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text = data["text"]

    if len(text) > 1500:
        return jsonify({"error": "Text too long. Please reduce input to under 1500 characters."}), 400

    try:
        summary = summarizer(text, max_length=120, min_length=30, do_sample=False)[0]["summary_text"]
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
