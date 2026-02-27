import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="Workplace Fatigue Detector", layout="wide")

st.title("🧠 Workplace Fatigue Detector")
st.markdown("Real-time face + body fatigue monitoring")

run = st.checkbox("Start Camera")

frame_placeholder = st.empty()
status_placeholder = st.empty()

# Proper MediaPipe initialization (IMPORTANT)
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

head_drop_counter = 0
movement_history = []
fatigue_score = 0


def calculate_head_drop(landmarks, img_h):
    nose = landmarks[1]
    chin = landmarks[152]
    return (chin.y - nose.y) * img_h


if run:
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh, mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera access denied or unavailable.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_results = face_mesh.process(rgb)
            pose_results = pose.process(rgb)

            img_h, img_w, _ = frame.shape

            # FACE PROCESSING
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS
                    )

                    drop_value = calculate_head_drop(
                        face_landmarks.landmark,
                        img_h
                    )

                    if drop_value > 80:
                        head_drop_counter += 1
                    else:
                        head_drop_counter = 0

            # POSE PROCESSING
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                left_shoulder = pose_results.pose_landmarks.landmark[11]
                right_shoulder = pose_results.pose_landmarks.landmark[12]

                shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
                movement_history.append(shoulder_diff)

                if len(movement_history) > 30:
                    movement_history.pop(0)

            # FATIGUE CALCULATION
            if len(movement_history) > 10:
                movement_variation = np.std(movement_history)

                if movement_variation < 0.01:
                    fatigue_score += 1
                else:
                    fatigue_score = max(0, fatigue_score - 1)

            if head_drop_counter > 15:
                fatigue_score += 2

            # ALERT SYSTEM
            if fatigue_score > 15:
                cv2.putText(
                    frame,
                    "⚠ FATIGUE DETECTED!",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )
                status_placeholder.error("⚠ Fatigue Detected!")
            else:
                status_placeholder.success("✅ Active")

            frame_placeholder.image(frame, channels="BGR")

    cap.release()
