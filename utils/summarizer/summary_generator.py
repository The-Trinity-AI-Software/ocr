# -*- coding: utf-8 -*-
"""
Created on Mon May 19 15:32:09 2025

@author: HP
"""

from transformers import pipeline

summary_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_license_summary(data: dict):
    name = f"{data.get('first_name', '')} {data.get('last_name', '')}".strip()
    dob = data.get("dob", "")
    dl = data.get("license_number", "")
    exp = data.get("expiry_date", "")
    cat = data.get("category", "")
    location = data.get("address", "")
    gender = data.get("gender", "")

    raw_text = f"{name} born on {dob}, gender {gender}, holds Driving License number {dl} in category {cat}, residing at {location}, valid till {exp}."

    summary = summary_pipeline(raw_text, max_length=60, min_length=20, do_sample=False)
    return summary[0]["summary_text"]