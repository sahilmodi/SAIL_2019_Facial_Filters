from model import CNNModel
import cv2
import numpy as np
from filter import Filter
import os

# Load the model built in the previous step
my_model = CNNModel('my_model.h5')

# TODO: Face cascade to detect faces
face_cascade = ...

# Load the webcam
camera = cv2.VideoCapture(0)

# TODO: add more filters!
moustache = Filter(os.path.join('filters', 'moustache.png'))

# Keep reading the camera
while True:
    # Grab the current paintWindow
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    frame2 = np.copy(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # TODO: Detect faces
    faces = ...
    if len(faces) == 0:
        cv2.putText(frame, "Finding Face...", (37, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Selfie Filters', frame)

    for (x, y, w, h) in faces:
        # Grab the face
        gray_face = gray[y:y+h, x:x+w]
        color_face = frame[y:y+h, x:x+w]

        # Normalize to match the input format of the model - Range of pixel to [0, 1]
        gray_normalized = gray_face / 255

        # Resize it to 96x96 to match the input format of the model
        original_shape = gray_face.shape
        face_resized = cv2.resize(gray_normalized, (96, 96), interpolation = cv2.INTER_AREA)
        face_resized_copy = face_resized.copy()
        face_resized = face_resized.reshape(1, 96, 96, 1)

        # Predicting the keypoints using the model
        keypoints = my_model.predict(face_resized)

        # De-Normalize the keypoints values
        keypoints = keypoints * 48 + 48

        # Map the Keypoints back to the original image
        face_resized_color = cv2.resize(color_face, (96, 96), interpolation = cv2.INTER_AREA)
        face_resized_color2 = np.copy(face_resized_color)

        x_scale = original_shape[1] / 96
        y_scale = original_shape[0] / 96
        
        # Pair them together
        points = []
        for i, co in enumerate(keypoints[0][0::2]):
            points.append((co, keypoints[0][1::2][i]))

        # Transform points to global coordinates
        for i in range(len(points)):
            points[i] *= np.array([x_scale, y_scale])
            points[i] += np.array([x, y])

        # TODO: Add more FILTERS to the frame
        moustache.nose_overlay(points, frame, -5)
        
        # Add KEYPOINTS to the frame2
        for keypoint in points:
            (px, py) = np.array(keypoint, dtype=np.int)
            cv2.circle(frame2, (px, py), 1, (0,255,0), 1)

        # Show the frame and the frame2
        cv2.imshow("Selfie Filters", frame)
        cv2.imshow("Keypoints", frame2)

    # If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
