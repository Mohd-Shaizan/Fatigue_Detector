import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import mediapipe as mp
import numpy as np

# 1. Page Config
st.set_page_config(page_title="Workplace Fatigue Detector", layout="wide")

st.title("🧠 Workplace Fatigue Detector")
st.markdown("Real-time Face + Body Monitoring")

# 2. Correct WebRTC Configuration
# For Streamlit Cloud/Remote access, you MUST have working STUN servers.
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# 3. MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class FatigueProcessor(VideoProcessorBase):
    def __init__(self):
        # State variables
        self.head_counter = 0
        self.fatigue_score = 0
        self.movement_history = []

        # Optimization: Lower complexity and set static_image_mode to False for tracking
        self.face_mesh = mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = mp_pose.Pose(
            model_complexity=0,  # 0 is fastest, better for real-time WebRTC
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        # Flip image for a "mirror" effect (optional but more natural)
        img = cv2.flip(img, 1)
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img.shape

        # Process landmarks
        face_results = self.face_mesh.process(rgb)
        pose_results = self.pose.process(rgb)

        # ---- FACE LOGIC ----
        if face_results.multi_face_landmarks:
            for landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    img, landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=1)
                )

                # Detection: Head drop
                nose = landmarks.landmark[1]
                chin = landmarks.landmark[152]
                drop = (chin.y - nose.y) * img_h

                if drop > 80:
                    self.head_counter += 1
                else:
                    self.head_counter = 0

        # ---- POSE LOGIC ----
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # Detect shoulder alignment variation
            left_shoulder = pose_results.pose_landmarks.landmark[11]
            right_shoulder = pose_results.pose_landmarks.landmark[12]
            
            diff = abs(left_shoulder.y - right_shoulder.y)
            self.movement_history.append(diff)

            if len(self.movement_history) > 20:
                self.movement_history.pop(0)

        # ---- FATIGUE CALCULATION ----
        if len(self.movement_history) > 10:
            variation = np.std(self.movement_history)
            # High variation means moving; low variation means slumped/still
            if variation < 0.005: 
                self.fatigue_score += 1
            else:
                self.fatigue_score = max(0, self.fatigue_score - 1)

        if self.head_counter > 15:
            self.fatigue_score += 2

        # ---- UI OVERLAY ----
        color = (0, 255, 0)
        status = "NORMAL"
        
        if self.fatigue_score > 15:
            status = "WARNING: FATIGUE"
            color = (0, 0, 255) # Red
            
        cv2.putText(img, f"Status: {status}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, f"Score: {int(self.fatigue_score)}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 4. App Execution
webrtc_streamer(
    key="fatigue-detector",
    video_processor_factory=FatigueProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True, # Helps with performance
)
