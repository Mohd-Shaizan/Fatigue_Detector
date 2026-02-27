import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import mediapipe as mp
import numpy as np
import os

# Disable GPU acceleration for MediaPipe to avoid EGL errors on headless servers
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

st.set_page_config(page_title="Workplace Fatigue Detector", layout="wide")

st.title("🧠 Workplace Fatigue Detector")
st.markdown("Real-time Face + Body Monitoring (CPU Optimized)")

# --- GLOBAL INITIALIZATION (Outside the class to fix permission/threading issues) ---
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Use model_complexity=0 (Lite) for Pose and static_image_mode=False for better performance
# We initialize them once globally
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pose = mp_pose.Pose(
    model_complexity=0, 
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
        img = cv2.flip(img, 1) # Mirror view
        img_h, img_w, _ = img.shape
        
        # Performance: MediaPipe works better with RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process landmarks using global instances
        face_results = face_mesh.process(rgb)
        pose_results = pose_results = pose.process(rgb)

        # 1. Face Logic
        if face_results.multi_face_landmarks:
            for landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(img, landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                
                # Head tilt detection
                nose = landmarks.landmark[1]
                chin = landmarks.landmark[152]
                if (chin.y - nose.y) * img_h > 80:
                    self.head_counter += 1
                else:
                    self.head_counter = 0

        # 2. Pose Logic
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(img, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            ls = pose_results.pose_landmarks.landmark[11]
            rs = pose_results.pose_landmarks.landmark[12]
            
            diff = abs(ls.y - rs.y)
            self.movement_history.append(diff)
            if len(self.movement_history) > 20:
                self.movement_history.pop(0)

        # 3. Fatigue Scoring
        if len(self.movement_history) > 10:
            variation = np.std(self.movement_history)
            if variation < 0.005: # Slumping/Stillness
                self.fatigue_score += 0.5
            else:
                self.fatigue_score = max(0, self.fatigue_score - 0.2)

        if self.head_counter > 15:
            self.fatigue_score += 2

        # UI Visuals
        alert = self.fatigue_score > 15
        label = "⚠️ FATIGUE ALERT" if alert else "NORMAL"
        color = (0, 0, 255) if alert else (0, 255, 0)
        
        cv2.putText(img, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(img, (0, 0), (img_w, img_h), color, 10 if alert else 0)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# STUN servers are mandatory for the video to start in Cloud
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="fatigue-main",
    video_processor_factory=FatigueProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True, # Critical for keeping the UI responsive
)
