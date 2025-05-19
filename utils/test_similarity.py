# -*- coding: utf-8 -*-
"""
Created on Mon May 12 13:37:27 2025

@author: HP
"""

import os
#os.environ["OMP_NUM_THREADS"] = "1"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity



app = FaceAnalysis(name='buffalo_l')  # ✅ includes detection
app.prepare(ctx_id=-1)

def compare_faces(face1_path, face2_path):
    face1_img = cv2.imread(face1_path)
    face2_img = cv2.imread(face2_path)

    face1_img = cv2.cvtColor(face1_img, cv2.COLOR_BGR2RGB)
    face2_img = cv2.cvtColor(face2_img, cv2.COLOR_BGR2RGB)

    faces1 = app.get(face1_img)
    faces2 = app.get(face2_img)

    if not faces1 or not faces2:
        print("❌ Face not detected in one of the images.")
        return 0.0

    emb1 = faces1[0].embedding
    emb2 = faces2[0].embedding
    return float(cosine_similarity([emb1], [emb2])[0][0])

