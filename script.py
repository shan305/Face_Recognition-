import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import os
print(cv2.__version__)
print(np.__version__)
def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        if not frames:
            print("Error: No frames were read from the video.")
        else:
            print("Frames successfully loaded.")

    return frames

def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

def extract_face_data(frames, face_cascade):
    face_data = []
    labels = []

    for i, frame in enumerate(frames):
        faces = detect_faces(frame, face_cascade)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face, (64, 64))
            face_data.append(resized_face)
            labels.append(1)  # Assuming positive label for faces

    print("Face data successfully extracted from frames.")
    return np.array(face_data), np.array(labels)

def train_model(X, y):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32)

    return model

def save_model(model, model_path):
    model.save(model_path)
    print("Model saved at {}".format(model_path))

def process_frames(frames, model):
    predictions = []

    for frame in frames:
        faces = detect_faces(frame, face_cascade)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face, (64, 64))
            prediction = model.predict(np.array([resized_face]))[0, 0]
            predictions.append(prediction)

    return predictions

def display_recognized_faces(frames, predictions, frames_to_display=2):
    recognized_faces = set()
    displayed_frames = 0

    for frame, prediction in zip(frames, predictions):
        faces = detect_faces(frame, face_cascade)

        if any(prediction > 0.5 for _ in faces):
            recognized_faces.clear()

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                resized_face = cv2.resize(face, (64, 64))
                prediction = model.predict(np.array([resized_face]))[0, 0]

                if prediction > 0.5:
                    recognized_faces.add((x, y, w, h))

            for (x, y, w, h) in recognized_faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.show()

            displayed_frames += 1
            if displayed_frames >= frames_to_display:
                break

def save_and_export(frames, predictions, model, output_directory, output_filename):
    output_path = os.path.join(output_directory, output_filename)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    print("Output Path:", output_path)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_video = cv2.VideoWriter(output_path, fourcc, 30, (frames[0].shape[1], frames[0].shape[0]))

    if not output_video.isOpened():
        print("Error: VideoWriter could not open the output file.")
    else:
        print("VideoWriter is open.")

        for frame, prediction in zip(frames, predictions):
            print("Processing a frame...")
            faces = detect_faces(frame, face_cascade)

            recognized_faces = []
            for (x, y, w, h), pred in zip(faces, predictions):
                face = frame[y:y+h, x:x+w]
                resized_face = cv2.resize(face, (64, 64))
                pred = model.predict(np.array([resized_face]))[0, 0]

                if pred > 0.5:
                    recognized_faces.append((x, y, w, h))

            for (x, y, w, h) in recognized_faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            label = "Face" if prediction > 0.5 else "No Face"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            success = output_video.write(frame)
            if not success:
                print("Error writing frame to video.")

        output_video.release()
        print("VideoWriter released.")

# Main script
video_path = 'video.mp4'
output_directory = 'output'
output_filename = 'output.mp4'
model_path = 'face_recognition_model.keras'
import os

if os.path.exists(video_path):
    print("Video file exists.")
else:
    print("Error: Video file not found.")
# Load video frames
frames = load_video_frames(video_path)

# Load face cascade classifier
face_cascade = load_face_cascade()

# Extract face data and train the model
X, y = extract_face_data(frames, face_cascade)
model = train_model(X, y)

# Save the trained model
save_model(model, model_path)

# Load the trained model
model = load_model(model_path)

# Process new video frames
video_path_new = 'video.mp4'
new_frames = load_video_frames(video_path_new)

# Process frames and make predictions
predictions_new = process_frames(new_frames, model)

# Display recognized faces in new frames
display_recognized_faces(new_frames, predictions_new)

# Save and export the output video
save_and_export(new_frames, predictions_new, model, output_directory, output_filename)
