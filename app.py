import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="VisionAid AI", layout="centered")

st.markdown("<h2 style='text-align: center; color: #00bcd4;'>👁️ VisionAid Assistant</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Real-time Detection & Voice Alerts</p>", unsafe_allow_html=True)

# 1. تحميل النموذج
@st.cache_resource
def load_model():
    return YOLO("yolo26n.pt")

model = load_model()

# الفئات المستهدفة
SMART_CLASSES = [0, 2, 3, 9, 56, 57, 59, 60, 61, 62, 68, 69, 71, 72]

# ملف لتخزين التنبيه اللحظي ومشاركته مع واجهة الجافاسكريبت
ALERT_FILE = "/tmp/visionaid_alert.txt"
with open(ALERT_FILE, "w") as f:
    f.write("clear")

class VideoProcessor(VideoTransformerBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        total_area = h * w

        # تطبيق الكشف
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
                if ratio > 0.07 and ratio > max_ratio:
                    max_ratio = ratio
                    closest_name = name

            if closest_name:
                if max_ratio > 0.22:
                    current_text = f"Warning, {closest_name} ahead"
                else:
                    current_text = f"{closest_name} ahead"
            else:
                current_text = "clear"
        else:
            current_text = "clear"

        # حفظ التنبيه مباشرة
        try:
            with open(ALERT_FILE, "w") as f:
                f.write(current_text)
        except Exception:
            pass

        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# إعداد الكاميرا
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

# 2. قراءة التنبيه الحالي في Streamlit
current_alert = "clear"
try:
    with open(ALERT_FILE, "r") as f:
        current_alert = f.read().strip()
except Exception:
    pass

# لوحة التحكم الصوتية المباشرة
st.components.v1.html(f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{ margin: 0; padding: 0; font-family: -apple-system, sans-serif; }}
  .voice-btn {{
    background-color: #00bcd4;
    color: white;
    border: none;
    padding: 12px;
    font-size: 15px;
    font-weight: bold;
    border-radius: 8px;
    cursor: pointer;
    width: 100%;
  }}
  .voice-btn.active {{
    background-color: #4caf50;
  }}
  #status-box {{
    margin-top: 8px;
    font-size: 14px;
    font-weight: bold;
    color: #333;
    text-align: center;
  }}
</style>
</head>
<body>

<button id="audio-btn" class="voice-btn" onclick="initAudio()">🔊 Click Here to Enable Sound</button>
<div id="status-box">Status: Waiting for click...</div>

<script>
  let synth = window.speechSynthesis;
  let audioEnabled = false;
  let targetVoice = null;
  let lastSpoken = "";
  let lastTime = 0;

  function setVoice() {{
    let voices = synth.getVoices();
    targetVoice = voices.find(v => (v.name.includes("Google") || v.name.includes("Samantha") || v.name.includes("Natural") || v.name.includes("Zira")) && v.lang.startsWith("en")) 
                  || voices.find(v => v.lang.startsWith("en"));
  }}
  setVoice();
  if (speechSynthesis.onvoiceschanged !== undefined) {{
    speechSynthesis.onvoiceschanged = setVoice;
  }}

  function initAudio() {{
    audioEnabled = true;
    let btn = document.getElementById("audio-btn");
    btn.textContent = "✅ Sound Active";
    btn.className = "voice-btn active";
    document.getElementById("status-box").textContent = "Active: Monitoring camera...";
    
    let u = new SpeechSynthesisUtterance("System online");
    if (targetVoice) u.voice = targetVoice;
    synth.speak(u);
  }}

  function speakAlert(text) {{
    if (!audioEnabled || !text || text === "clear" || synth.speaking) return;
    let now = Date.now();
    if (now - lastTime < 2500) return;

    let utter = new SpeechSynthesisUtterance(text);
    if (targetVoice) utter.voice = targetVoice;
    utter.rate = 1.05;
    utter.pitch = 1.1;
    synth.speak(utter);
    lastTime = now;
  }}

  // قيمة التنبيه المحقونة مباشرة من بايثون
  let alertFromPython = "{current_alert}";

  if (alertFromPython && alertFromPython !== "clear") {{
    document.getElementById("status-box").textContent = "Alert: " + alertFromPython;
    speakAlert(alertFromPython);
  }}
</script>

</body>
</html>
""", height=110)
