import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Suppress oneDNN warning

from flask import Flask, request, render_template, send_file, send_from_directory
import json, csv
from utils.ocr_utils import extract_text_with_llm
from utils.aws_utils import download_single_image_from_s3
from flask_cors import CORS
app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = "replace_with_secure_random_string"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('output', exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    status = None
    ocr_image_path = None

    if request.method == "POST":
        s3_uri = request.form.get("doc_s3_uri")
        threshold = float(request.form.get("threshold", 0.5))

        try:
            # 🔒 Uses hardcoded AWS keys inside utils/aws_utils.py
            downloaded = download_single_image_from_s3(
                s3_uri, app.config['UPLOAD_FOLDER']
            )
            image_path = downloaded[0]
            ocr_image_path = os.path.relpath(image_path, "static").replace("\\", "/")

            # 🧠 Run OCR + NER + Summary
            result = extract_text_with_llm(image_path)
            result["ocr_text"] = result.pop("full_text", "")
            result["threshold"] = threshold

            # 💾 Save results
            with open("output/result.json", "w") as jf:
                json.dump(result, jf, indent=4)
            with open("output/result.csv", "w", newline="") as cf:
                writer = csv.DictWriter(cf, fieldnames=result.keys())
                writer.writeheader()
                writer.writerow(result)

            status = f"✅ OCR completed from file: {os.path.basename(image_path)}"

        except Exception as e:
            status = f"❌ Error during OCR: {str(e)}"

    return render_template("index.html", result=result, connection_status=status, ocr_image_path=ocr_image_path)


@app.route("/download/json")
def download_json():
    return send_file("output/result.json", as_attachment=True)

@app.route("/download/csv")
def download_csv():
    return send_file("output/result.csv", as_attachment=True)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=9098)
