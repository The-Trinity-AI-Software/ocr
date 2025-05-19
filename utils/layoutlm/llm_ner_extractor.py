# -*- coding: utf-8 -*-
"""
Created on Mon May 19 15:30:56 2025

@author: HP
"""

from transformers import pipeline

# Load once
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# Field label mapping for clarity
ENTITY_LABELS = {
    "PER": "name",
    "LOC": "location",
    "ORG": "organization",
    "MISC": "other"
}

def extract_entities_from_text(text):
    entities = ner_pipeline(text)
    structured_data = {
        "first_name": "",
        "last_name": "",
        "license_number": "",
        "dob": "",
        "gender": "",
        "issue_date": "",
        "expiry_date": "",
        "address": "",
        "category": "",
        "nationality": "USA"
    }

    # Heuristics + NER
    names = [ent["word"] for ent in entities if ent["entity_group"] == "PER"]
    if names:
        structured_data["first_name"] = names[0]
        if len(names) > 1:
            structured_data["last_name"] = names[-1]

    # License number and DOB pattern fallback
    import re
    structured_data["license_number"] = extract_by_regex(r'DL[:\\-\\s]*([A-Z0-9]+)', text)
    structured_data["dob"] = extract_by_regex(r'DOB[:\\-\\s]*([0-9/]+)', text)
    structured_data["issue_date"] = extract_by_regex(r'ISS[:\\-\\s]*([0-9/]+)', text)
    structured_data["expiry_date"] = extract_by_regex(r'EXP[:\\-\\s]*([0-9/]+)', text)
    structured_data["gender"] = extract_by_regex(r'SEX[:\\-\\s]*(M|F|Male|Female)', text)
    structured_data["category"] = extract_by_regex(r'CLASS[:\\-\\s]*([A-Z])', text)
    structured_data["address"] = extract_by_regex(r'([0-9]+ .*?, CA \\d+)', text)

    return structured_data

def extract_by_regex(pattern, text, default=""):
    import re
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else default