import cv2

def detect_face(image_path_or_array):
    if isinstance(image_path_or_array, str):
        image = cv2.imread(image_path_or_array)
    else:
        image = image_path_or_array

    if image is None:
        print("❌ Failed to load image")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("❌ No face detected")
        return None

    x, y, w, h = faces[0]
    face_crop = image[y:y+h, x:x+w]
    return face_crop
