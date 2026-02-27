import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="Workplace Fatigue Detector", layout="wide")

# ---- GLOBALS (Initialize outside the class to avoid repeated downloads) ----
# This ensures MediaPipe loads once during app startup
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# We initialize these globally so they are ready before the WebRTC thread starts
face_mesh_instance = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# Model complexity 1 is usually pre-packaged and avoids the "lite" download trigger
pose_instance = mp_pose.Pose(
    model_complexity=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class FatigueProcessor(VideoProcessorBase):
    def __init__(self):
        self.head_counter = 0
        self.fatigue_score = 0
        self.movement_history = []

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img.shape

        # Use the global instances instead of local ones
        face_results = face_mesh_instance.process(rgb)
        pose_results = pose_instance.process(rgb)

        # ---- FACE DETECTION ----
        if face_results.multi_face_landmarks:
            for landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(img, landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                
                nose = landmarks.landmark[1]
                chin = landmarks.landmark[152]
                drop = (chin.y - nose.y) * img_h
                if drop > 80:
                    self.head_counter += 1
                else:
                    self.head_counter = 0

        # ---- POSE DETECTION ----
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(img, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            ls = pose_results.pose_landmarks.landmark[11]
            rs = pose_results.pose_landmarks.landmark[12]
            
            diff = abs(ls.y - rs.y)
            self.movement_history.append(diff)
            if len(self.movement_history) > 20:
                self.movement_history.pop(0)

        # ---- FATIGUE LOGIC ----
        if len(self.movement_history) > 10:
            variation = np.std(self.movement_history)
            if variation < 0.005:
                self.fatigue_score += 1
            else:
                self.fatigue_score = max(0, self.fatigue_score - 1)

        if self.head_counter > 15:
            self.fatigue_score += 2

        # Status Overlay
        status = "FATIGUE DETECTED" if self.fatigue_score > 15 else "NORMAL"
        color = (0, 0, 255) if self.fatigue_score > 15 else (0, 255, 0)
        cv2.putText(img, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC Config
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="fatigue-detector",
    video_processor_factory=FatigueProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
