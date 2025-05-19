# similarity_utils.py
import os
import cv2
import boto3
from urllib.parse import urlparse
import numpy as np
from deepface import DeepFace
from pdf2image import convert_from_path
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#face_model = FaceAnalysis(name='buffalo_l')
face_model = FaceAnalysis(name='deepface')
face_model.prepare(ctx_id=-1)

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def parse_s3_uri(s3_uri):
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    parts = s3_uri.replace("s3://", "").split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URI structure: {s3_uri}")
    return parts[0], parts[1]

def download_single_image_from_s3(access_key, secret_key, s3_uri, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    bucket, key = parse_s3_uri(s3_uri)
    filename = os.path.basename(key)
    local_path = os.path.join(local_dir, filename)
    s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    s3.download_file(bucket, key, local_path)
    return [local_path]

def download_s3_folder(access_key, secret_key, s3_uri, local_dir):
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')
    s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' not in response:
        raise Exception("No files found in folder")
    os.makedirs(local_dir, exist_ok=True)
    downloaded = []
    for obj in response['Contents']:
        key = obj['Key']
        filename = os.path.basename(key)
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf')):
            continue
        local_path = os.path.join(local_dir, filename)
        s3.download_file(bucket, key, local_path)
        downloaded.append(local_path)
    return downloaded

def convert_pdf_to_images(pdf_path, output_folder):
    pages = convert_from_path(pdf_path)
    image_paths = []
    for i, page in enumerate(pages):
        out_path = os.path.join(output_folder, f"page_{i+1}.jpg")
        page.save(out_path, "JPEG")
        image_paths.append(out_path)
    return image_paths

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

import cv2

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face(image_path, save_path=None, overlay_path=None):
    print(f"📥 Loading image: {image_path}")
    image = cv2.imread(image_path) if isinstance(image_path, str) else image_path
    if image is None:
        print("❌ Failed to read image.")
        return None

    try:
        h, w = image.shape[:2]
        if h < 300 or w < 300:
            image = cv2.resize(image, (300, 300))

        # CLAHE enhancement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        image_clahe = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        rgb = cv2.cvtColor(image_clahe, cv2.COLOR_BGR2RGB)
        faces = face_model.get(rgb)

        face_crop = None
        if faces:
            print("✅ Face detected by InsightFace.")
            bbox = faces[0].bbox.astype(int)
            x1, y1, x2, y2 = bbox
            face_crop = image[y1:y2, x1:x2]
        else:
            print("⚠️ InsightFace failed. Trying Haar Cascade...")
            detected = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(detected) > 0:
                x, y, w, h = detected[0]
                face_crop = image[y:y+h, x:x+w]

        if face_crop is not None and face_crop.size > 0:
            if save_path:
                cv2.imwrite(save_path, face_crop)
                print(f"✅ Face crop saved at: {save_path}")

            if overlay_path:
                overlay = image.copy()
                if faces:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                elif len(detected) > 0:
                    cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.imwrite(overlay_path, overlay)
                print(f"🖼 Overlay saved at: {overlay_path}")
        else:
            print("❌ No face detected after all attempts.")

        return face_crop

    except Exception as e:
        print(f"❌ Exception during face detection: {e}")
        return None



def compare_faces(face1_img, face2_img):
    if face1_img is None or face2_img is None:
        print("❌ One or both face images are None.")
        return 0.0
    try:
        emb1 = face_model.get(cv2.cvtColor(face1_img, cv2.COLOR_BGR2RGB))[0].embedding
        emb2 = face_model.get(cv2.cvtColor(face2_img, cv2.COLOR_BGR2RGB))[0].embedding
        score = float(cosine_similarity([emb1], [emb2])[0][0])
        print(f"📏 Cosine Similarity Score: {score}")
        return score
    except Exception as e:
        print(f"❌ Face embedding failed: {e}")
        return 0.0
