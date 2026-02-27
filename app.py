import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="Workplace Fatigue Detector", layout="wide")

st.title("🧠 Workplace Fatigue Detector")
st.markdown("Real-time Face + Body Monitoring")

# ---- WebRTC Configuration (VERY IMPORTANT FOR CLOUD) ----
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
        # Add more if deploying to production
    ]}
)

# ---- MediaPipe Setup ----
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class FatigueProcessor(VideoProcessorBase):
    def __init__(self):
        self.head_counter = 0
        self.fatigue_score = 0
        self.movement_history = []

        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.pose = mp_pose.Pose()

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img.shape

        face_results = self.face_mesh.process(rgb)
        pose_results = self.pose.process(rgb)

        # ---- FACE DETECTION ----
        if face_results.multi_face_landmarks:
            for landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS
                )

                nose = landmarks.landmark[1]
                chin = landmarks.landmark[152]
                drop = (chin.y - nose.y) * img_h

                if drop > 80:
                    self.head_counter += 1
                else:
                    self.head_counter = 0

        # ---- POSE DETECTION ----
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            left_shoulder = pose_results.pose_landmarks.landmark[11]
            right_shoulder = pose_results.pose_landmarks.landmark[12]

            diff = abs(left_shoulder.y - right_shoulder.y)
            self.movement_history.append(diff)

            if len(self.movement_history) > 20:
                self.movement_history.pop(0)

        # ---- FATIGUE LOGIC ----
        if len(self.movement_history) > 10:
            variation = np.std(self.movement_history)

            if variation < 0.01:
                self.fatigue_score += 1
            else:
                self.fatigue_score = max(0, self.fatigue_score - 1)

        if self.head_counter > 15:
            self.fatigue_score += 2

        # ---- ALERT ----
        if self.fatigue_score > 15:
            cv2.putText(
                img,
                "⚠ FATIGUE DETECTED",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # your mediapipe processing here

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="fatigue-detector",
    video_processor_factory=VideoProcessor,   # NEW API
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    rtc_configuration={
        "iceServers": []   # 🔥 IMPORTANT: disable STUN
    },
)
