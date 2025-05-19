from paddleocr import PaddleOCR
import os
import cv2
import re
from utils.layoutlm.llm_ner_extractor import extract_entities_from_text
from utils.summarizer.summary_generator import generate_license_summary

ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

def extract_text_with_llm(image_path):
    default_result = {
        "ocr_text": "OCR failed.",
        "first_name": "", "last_name": "",
        "license_number": "", "dob": "",
        "issue_date": "", "expiry_date": "",
        "address": "", "gender": "",
        "category": "", "nationality": "USA",
        "summary": ""
    }

    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        default_result["ocr_text"] = "Image not found."
        return default_result

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Cannot read image: {image_path}")
        default_result["ocr_text"] = "Unreadable image."
        return default_result

    try:
        result = ocr_model.ocr(image_path, cls=True)
        if not result or not result[0]:
            default_result["ocr_text"] = "No text found."
            return default_result

        full_text = " ".join([line[1][0] for line in result[0]])
        structured_data = extract_entities_from_text(full_text)
        structured_data["ocr_text"] = full_text

        try:
            structured_data["summary"] = generate_license_summary(structured_data)
        except Exception as e:
            print("⚠️ Failed to generate summary:", e)
            structured_data["summary"] = ""

        return structured_data

    except Exception as e:
        print(f"❌ OCR Exception: {e}")
        return default_result