import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# ==============================
# CONFIG
# ==============================
# We track the last 5 seconds to see if the user has stopped moving (slumping)
WINDOW_DURATION = 5 
FATIGUE_THRESHOLD = 60.0 # Risk index percentage to trigger alert

rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="GuardianEye Fatigue Monitor", layout="wide")
st.title("Workplace Fatigue Monitor")

st.markdown("""
This system monitors posture and movement frequency to detect signs of exhaustion.
- **Head Drop:** Detected via facial landmark displacement.
- **Micro-Movement:** Analyzes if the user has become unnaturally still (hypomotility).
""")
st.markdown("""
    <style>
    /* This targets the iframe containing the webcam */
    .element-container iframe {
        width: 600px !important;
        height: 450px !important;
        display: block;
        margin-left: auto;
        margin-right: auto;
        border-radius: 10px;
        border: 2px solid #00ffc8; /* Optional neon border */
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================
# VIDEO PROCESSOR
# ==============================
class FatigueProcessor(VideoProcessorBase):
    def __init__(self):
        # Using FaceMesh for precise head tilt and Pose for posture
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # Temporal buffers
        self.timestamps = deque()
        self.head_y_positions = deque() # Tracking vertical head movement
        self.motion_intensity = deque(maxlen=100) # Tracking general body movement

    def compute_fatigue_metrics(self):
        if len(self.head_y_positions) < 10:
            return 0, 0
        
        # 1. Head Drop Score (Vertical displacement)
        # Check if the current head position is significantly lower than the average
        head_array = np.array(self.head_y_positions)
        current_drop = head_array[-1] - np.mean(head_array)
        drop_score = max(0, current_drop * 500) # Scaled for visibility

        # 2. Movement Score (Stillness/Slumping)
        # Low standard deviation in movement indicates a "frozen" or fatigued state
        movement_std = np.std(head_array)
        stillness_score = max(0, 1 - (movement_std * 100))
        
        # Combined Risk Calculation
        risk_index = (drop_score * 0.6) + (stillness_score * 40)
        return round(min(risk_index, 100), 1), stillness_score

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        current_time = time.time()
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # Track Nose Bridge (Landmark 1)
                nose = landmarks.landmark[1]
                self.head_y_positions.append(nose.y)
                self.timestamps.append(current_time)

                # Draw simplified mesh for feedback
                for connection in self.mp_face_mesh.FACEMESH_CONTOURS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    p1 = landmarks.landmark[start_idx]
                    p2 = landmarks.landmark[end_idx]
                    cv2.line(img, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 255, 100), 1)

        # Maintain 5-second sliding window
        while self.timestamps and (current_time - self.timestamps[0]) > WINDOW_DURATION:
            self.timestamps.popleft()
            self.head_y_positions.popleft()

        risk, stillness = self.compute_fatigue_metrics()

        # UI Overlay Logic
        overlay = img.copy()
        panel_color = (20, 20, 20)
        if risk > FATIGUE_THRESHOLD:
            panel_color = (0, 0, 180) # Turn panel red on alert
            cv2.putText(img, "!!! FATIGUE ALERT !!!", (w//2 - 150, h - 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

        cv2.rectangle(overlay, (20, 20), (400, 180), panel_color, -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # Status Text
        neon_green = (0, 255, 180)
        cv2.putText(img, "GUARDIAN EYE v1.0", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, neon_green, 2)
        cv2.putText(img, f"Fatigue Risk: {risk}%", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Visual Progress Bar
        bar_width = int(risk * 3.4)
        cv2.rectangle(img, (40, 130), (380, 150), (50, 50, 50), -1)
        cv2.rectangle(img, (40, 130), (40 + bar_width, 150), neon_green, -1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==============================
# WEBRTC STREAM
# ==============================
# Create three columns; the camera will live in the middle one
col1, col2, col3 = st.columns([1, 2, 1]) 

with col2:
    webrtc_streamer(
        key="fatigue-monitor",
        video_processor_factory=FatigueProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

webrtc_streamer(
    key="fatigue-monitor",
    video_processor_factory=FatigueProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={
    "video": {
        "width": {"ideal": 940},
        "height": {"ideal": 580},
        "frameRate": {"ideal": 60}
    },
    "audio": False,
},
)
