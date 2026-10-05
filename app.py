import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import av
from ultralytics import YOLO

st.set_page_config(page_title="VisionAid AI", layout="centered")

st.markdown("<h2 style='text-align: center; color: #00bcd4;'>👁️ VisionAid Assistant</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Real-time Detection & Voice Alerts</p>", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO("yolo26n.pt")

model = load_model()


SMART_CLASSES = [0, 2, 3, 9, 56, 57, 59, 60, 61, 62, 68, 69, 71, 72]


class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.latest_alert = ""

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        total_area = h * w

       
        results = model.predict(
            source=img,
            imgsz=320,
            conf=0.35,
            classes=SMART_CLASSES,
            verbose=False
        )
        result = results[0]
        annotated_img = result.plot()

        
        max_ratio = 0.0
        closest_name = None
        if len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                name = model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                ratio = ((x2 - x1) * (y2 - y1)) / total_area
                if ratio > 0.08 and ratio > max_ratio:
                    max_ratio = ratio
                    closest_name = name

            if closest_name:
                if max_ratio > 0.22:
                    self.latest_alert = f"Warning, {closest_name} ahead"
                else:
                    self.latest_alert = f"{closest_name} detected"

        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

ctx = webrtc_streamer(
    key="visionaid",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    async_processing=True,
)
st.components.v1.html("""
<script>
    let synth = window.speechSynthesis;
    let lastText = "";
    let lastTime = 0;

    function speak(text) {
        if (!text || synth.speaking || Date.now() - lastTime < 2500) return;
        let utter = new SpeechSynthesisUtterance(text);
        utter.rate = 1.05;
        utter.pitch = 1.1;
        synth.speak(utter);
        lastTime = Date.now();
    }

    // فحص دوري لقراءة التنبيهات
    setInterval(() => {
        const textElements = window.parent.document.querySelectorAll("p, div, span");
        for (let el of textElements) {
            if (el.innerText.includes("ahead") || el.innerText.includes("detected")) {
                if (el.innerText !== lastText) {
                    lastText = el.innerText;
                    speak(lastText);
                }
                break;
            }
        }
    }, 500);
</script>
""", height=0)
