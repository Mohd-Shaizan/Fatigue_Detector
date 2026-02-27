import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

st.set_page_config(page_title="Workplace Fatigue Detector", layout="wide")

st.title("🧠 Workplace Fatigue Detector")
st.markdown("Detects head nodding, posture changes and reduced movement.")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Fatigue tracking variables
head_drop_counter = 0
movement_history = []
fatigue_score = 0

def calculate_head_angle(landmarks, image_w, image_h):
    nose = landmarks[1]
    chin = landmarks[152]

    nose_y = nose.y * image_h
    chin_y = chin.y * image_h

    return chin_y - nose_y

if run:
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh, \
         mp_pose.Pose() as pose:

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not working")
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_face = face_mesh.process(image_rgb)
            results_pose = pose.process(image_rgb)

            image_h, image_w, _ = frame.shape

            # FACE DETECTION
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS
                    )

                    angle = calculate_head_angle(
                        face_landmarks.landmark,
                        image_w,
                        image_h
                    )

                    if angle > 80:  # head dropped
                        head_drop_counter += 1
                    else:
                        head_drop_counter = 0

            # BODY DETECTION
            if results_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                left_shoulder = results_pose.pose_landmarks.landmark[11]
                right_shoulder = results_pose.pose_landmarks.landmark[12]

                shoulder_movement = abs(left_shoulder.y - right_shoulder.y)
                movement_history.append(shoulder_movement)

                if len(movement_history) > 30:
                    movement_history.pop(0)

            # FATIGUE LOGIC
            if len(movement_history) > 10:
                movement_variation = np.std(movement_history)

                if movement_variation < 0.01:
                    fatigue_score += 1
                else:
                    fatigue_score = max(0, fatigue_score - 1)

            if head_drop_counter > 15:
                fatigue_score += 2

            # ALERT
            if fatigue_score > 15:
                cv2.putText(frame,
                            "⚠ FATIGUE DETECTED!",
                            (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 0, 255),
                            4)

            FRAME_WINDOW.image(frame, channels="BGR")

        cap.release()
