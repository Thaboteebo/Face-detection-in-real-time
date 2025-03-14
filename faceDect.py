import face_recognition as fr
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Paths to the pre-trained models
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# Function to load known faces
def get_encoded_faces():
    encoded = {}
    dataset_dir = "dataset"  # Local folder for known faces
    for file_name in os.listdir(dataset_dir):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            # Load the image and encode the face
            face = fr.load_image_file(f"{dataset_dir}/{file_name}")
            encodings = fr.face_encodings(face)
            if encodings:
                encoded[file_name.split(".")[0]] = encodings[0]
    return encoded

# Function to classify faces in a test image
def classify_face(image_path):
    # Load known faces
    known_faces = get_encoded_faces()
    known_face_names = list(known_faces.keys())
    known_face_encodings = list(known_faces.values())

    # Load the test image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    # Convert the image to RGB format
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = fr.face_locations(rgb_img)

    if not face_locations:
        print("No faces detected in the image.")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        return []

    # Encode unknown faces
    unknown_encodings = fr.face_encodings(rgb_img, face_locations)
    face_names = []

    # Compare faces
    for face_encoding in unknown_encodings:
        matches = fr.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        if True in matches:
            best_match_index = np.argmin(fr.face_distance(known_face_encodings, face_encoding))
            name = known_face_names[best_match_index]
        face_names.append(name)

    # Draw bounding boxes and display names
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img, name, (left + 6, bottom - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display the image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    print("Recognized faces:", face_names)
    return face_names

# Run the face recognition system
if __name__ == "__main__":
    # Path to the test image
    image_path = "test2.jpg"
    print(classify_face(image_path))
