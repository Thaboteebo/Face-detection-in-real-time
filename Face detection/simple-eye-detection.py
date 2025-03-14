import cv2
import numpy as np
import time
from collections import deque
import winsound

# Load the pre-trained face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Constants for sound alerts
ALERT_FREQUENCY = 1000  # Hz
ALERT_DURATION = 1000   # milliseconds
ALERT_INTERVAL = 3.0   # seconds between alerts

# Initialize variables for fatigue detection
class FatigueDetector:
    def __init__(self):
        self.blink_counter = 0
        self.start_time = time.time()
        self.last_blink_time = time.time()
        self.last_alert_time = time.time()
        self.eyes_closed_start = None
        self.total_closed_time = 0
        self.closed_durations = deque(maxlen=30)  # Store last 30 seconds of eye closure durations
        self.blinks_per_minute = 0
        self.perclos = 0  # Percentage of eye closure
        self.fatigue_level = "Normal"
        self.eye_status = "Open"
        self.frame_count = 0
        self.last_minute_blinks = deque(maxlen=60)  # Store blinks for last 60 seconds
        self.previous_fatigue_level = "Normal"

    def update_eye_status(self, eyes_detected):
        current_time = time.time()
        
        if not eyes_detected:  # Eyes are closed
            if self.eye_status == "Open":  # Just closed
                self.eyes_closed_start = current_time
                self.eye_status = "Closed"
        else:  # Eyes are open
            if self.eye_status == "Closed":  # Just opened - complete a blink
                if self.eyes_closed_start is not None:
                    closed_duration = current_time - self.eyes_closed_start
                    if 0.1 <= closed_duration <= 0.5:  # Valid blink duration
                        self.blink_counter += 1
                        self.closed_durations.append(closed_duration)
                        self.last_minute_blinks.append(current_time)
                    self.total_closed_time += closed_duration
                self.eye_status = "Open"

    def should_play_alert(self, current_time):
        # Play alert if:
        # 1. Fatigue level changed to High/Mild from Normal
        # 2. Enough time has passed since last alert
        # 3. Fatigue is still detected
        time_since_last_alert = current_time - self.last_alert_time
        fatigue_worsened = (self.fatigue_level != "Normal" and 
                           self.previous_fatigue_level == "Normal")
        
        if (fatigue_worsened or 
            (self.fatigue_level != "Normal" and time_since_last_alert >= ALERT_INTERVAL)):
            self.last_alert_time = current_time
            return True
        return False

    def update_metrics(self):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        self.frame_count += 1

        # Update every second
        if elapsed_time >= 1:
            # Calculate PERCLOS (percentage of eye closure over time)
            self.perclos = (self.total_closed_time / elapsed_time) * 100
            
            # Calculate recent blink rate (blinks in last 60 seconds)
            recent_blinks = sum(1 for t in self.last_minute_blinks if current_time - t <= 60)
            self.blinks_per_minute = recent_blinks

            # Calculate average blink duration
            avg_blink_duration = np.mean(list(self.closed_durations)) if self.closed_durations else 0

            # Store previous fatigue level for alert detection
            self.previous_fatigue_level = self.fatigue_level

            # Update fatigue level based on multiple metrics
            if self.perclos > 20 or (self.blinks_per_minute < 10 and elapsed_time > 60):
                self.fatigue_level = "High Fatigue"
            elif (15 <= self.perclos <= 20 or 
                  (10 <= self.blinks_per_minute < 12 and elapsed_time > 60) or 
                  (avg_blink_duration > 0.3 if self.closed_durations else False)):
                self.fatigue_level = "Mild Fatigue"
            else:
                self.fatigue_level = "Normal"

            # Play alert sound if needed
            if self.should_play_alert(current_time):
                # Use a higher frequency for high fatigue
                freq = ALERT_FREQUENCY * 2 if self.fatigue_level == "High Fatigue" else ALERT_FREQUENCY
                winsound.Beep(freq, ALERT_DURATION)

            # Reset timing counters
            self.start_time = current_time
            self.total_closed_time = 0

    def get_display_info(self):
        return {
            'blinks': self.blink_counter,
            'bpm': self.blinks_per_minute,
            'perclos': self.perclos,
            'status': self.fatigue_level,
            'eye_status': self.eye_status
        }

# Initialize detector
detector = FatigueDetector()

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_detected = False
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        eyes_detected = len(eyes) > 0
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Update fatigue detector
    detector.update_eye_status(eyes_detected)
    detector.update_metrics()
    info = detector.get_display_info()

    # Display information on frame
    cv2.putText(frame, f"Blinks: {info['blinks']}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Blinks/min: {info['bpm']:.1f}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"PERCLOS: {info['perclos']:.1f}%", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Eye Status: {info['eye_status']}", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Fatigue: {info['status']}", (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if info['status'] != "Normal" else (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Fatigue Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
