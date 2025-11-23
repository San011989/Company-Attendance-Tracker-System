import os
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis

# Path where registered faces are stored
FACE_DIR = "registered_faces"  # folder where images from registration are saved
OUTPUT_CSV = "known_embeddings.csv"

# Initialize InsightFace model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# Storage for embeddings
data = []

# Loop through each registered person's folder
for person_name in os.listdir(FACE_DIR):
    person_path = os.path.join(FACE_DIR, person_name)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Could not read image: {img_path}")
            continue

        faces = app.get(img)
        if len(faces) == 0:
            print(f"⚠️ No face detected in: {img_path}")
            continue

        # Take the first face's embedding
        embedding = faces[0].embedding

        # Save the name + embedding
        row = [person_name] + embedding.tolist()
        data.append(row)

# Create dataframe
if len(data) == 0:
    print("❌ No embeddings generated. Please check your registered_faces folder.")
else:
    columns = ["name"] + [f"f{i}" for i in range(len(data[0]) - 1)]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved embeddings to {OUTPUT_CSV}")
