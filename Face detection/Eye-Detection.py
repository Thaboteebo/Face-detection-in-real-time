import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# Load the pre-trained face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file

# Indices for left and right eyes (from 68 facial landmarks)
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# Function to compute the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])  # Vertical line 1
    B = distance.euclidean(eye_points[2], eye_points[4])  # Vertical line 2
    C = distance.euclidean(eye_points[0], eye_points[3])  # Horizontal line
    
    ear = (A + B) / (2.0 * C)
    return ear

# Define EAR threshold and frame count for closed eyes
EAR_THRESHOLD = 0.4  # Lower means more sensitive
CLOSED_EYE_FRAME_COUNT = 5  # Number of frames for eyes closed detection
frame_counter = 0

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = detector(gray)  # Detect faces

    for face in faces:
        landmarks = predictor(gray, face)  # Get landmarks

        # Extract eye coordinates
        left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE])
        right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE])

        # Compute EAR for both eyes
        left_EAR = eye_aspect_ratio(left_eye_pts)
        right_EAR = eye_aspect_ratio(right_eye_pts)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        # Draw eyes
        cv2.polylines(frame, [left_eye_pts], isClosed=True, color=(0, 255, 0), thickness=1)
        cv2.polylines(frame, [right_eye_pts], isClosed=True, color=(0, 255, 0), thickness=1)

        # Check if eyes are closed
        if avg_EAR < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= CLOSED_EYE_FRAME_COUNT:
                cv2.putText(frame, "EYES CLOSED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            frame_counter = 0

    # Show the frame
    cv2.imshow("Live Eye Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
