import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="Workplace Fatigue Detector", layout="wide")

st.title("🧠 Workplace Fatigue Detector")
st.markdown("Real-time Face + Body Monitoring (Cloud Compatible)")

mp_face = mp.tasks.vision.FaceLandmarker
mp_pose = mp.tasks.vision.PoseLandmarker
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Load models (automatically downloaded)
face_options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=None),
    running_mode=VisionRunningMode.LIVE_STREAM,
)

pose_options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=None),
    running_mode=VisionRunningMode.LIVE_STREAM,
)


class FatigueDetector(VideoTransformerBase):
    def __init__(self):
        self.head_counter = 0
        self.fatigue_score = 0
        self.prev_shoulder_y = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_h, img_w, _ = img.shape

        # Fallback simple detection (since Tasks API model auto path is tricky in cloud)
        mp_face_mesh = mp.solutions.face_mesh
        mp_pose_solution = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils

        with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh, \
             mp_pose_solution.Pose() as pose:

            face_results = face_mesh.process(rgb)
            pose_results = pose.process(rgb)

            # FACE
            if face_results.multi_face_landmarks:
                for landmarks in face_results.multi_face_landmarks:
                    mp_draw.draw_landmarks(
                        img, landmarks, mp_face_mesh.FACEMESH_CONTOURS
                    )

                    nose = landmarks.landmark[1]
                    chin = landmarks.landmark[152]
                    drop = (chin.y - nose.y) * img_h

                    if drop > 80:
                        self.head_counter += 1
                    else:
                        self.head_counter = 0

            # POSE
            if pose_results.pose_landmarks:
                mp_draw.draw_landmarks(
                    img,
                    pose_results.pose_landmarks,
                    mp_pose_solution.POSE_CONNECTIONS,
                )

                left_shoulder = pose_results.pose_landmarks.landmark[11]
                right_shoulder = pose_results.pose_landmarks.landmark[12]

                diff = abs(left_shoulder.y - right_shoulder.y)
                self.prev_shoulder_y.append(diff)

                if len(self.prev_shoulder_y) > 20:
                    self.prev_shoulder_y.pop(0)

            # FATIGUE LOGIC
            if len(self.prev_shoulder_y) > 10:
                variation = np.std(self.prev_shoulder_y)

                if variation < 0.01:
                    self.fatigue_score += 1
                else:
                    self.fatigue_score = max(0, self.fatigue_score - 1)

            if self.head_counter > 15:
                self.fatigue_score += 2

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

        return img


webrtc_streamer(
    key="fatigue",
    video_transformer_factory=FatigueDetector,
    media_stream_constraints={"video": True, "audio": False},
)
