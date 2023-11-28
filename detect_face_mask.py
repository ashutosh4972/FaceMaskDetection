import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the pre-trained face mask detection model
model = load_model('saved_model.h5')

# Load the haarcascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video from the default camera (change the argument to the video file path if needed)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y + h, x:x + w]

        # Preprocess the face for the model
        face = cv2.resize(face, (128, 128))  # Resize the face image to match the model's input size
        face = preprocess_input(face)  # Apply preprocessing specific to the chosen model

        # Make predictions using the model
        predictions = model.predict(np.expand_dims(face, axis=0))

        # Get the label and confidence
        label = "Mask" if predictions[0][0] > 0.5 else "No Mask"
        confidence = predictions[0][0] if label == "Mask" else 1 - predictions[0][0]

        # Display the label and confidence on the frame
        label_text = f"{label}: {confidence:.2f}"
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if label == "Mask" else (0, 0, 255), 2)
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if label == "Mask" else (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face Mask Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
