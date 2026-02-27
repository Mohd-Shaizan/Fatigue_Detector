import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import mediapipe as mp
# Direct imports to bypass the 'solutions' attribute error
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
import numpy as np
import os

os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

st.set_page_config(page_title="Workplace Fatigue Detector", layout="wide")

st.title("🧠 Workplace Fatigue Detector")

# --- Initialize Global Instances ---
# We use the direct imports here
face_mesh_instance = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose_instance = mp_pose.Pose(
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
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img.shape

        # Process
        face_results = face_mesh_instance.process(rgb)
        pose_results = pose_instance.process(rgb)

        if face_results.multi_face_landmarks:
            for landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(img, landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                # (Detection logic remains the same...)

        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(img, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Final Streamer Call
webrtc_streamer(
    key="fatigue-check",
    video_processor_factory=FatigueProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
