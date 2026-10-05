import av
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoTransformerBase, webrtc_streamer
from ultralytics import YOLO

st.set_page_config(page_title="VisionAid AI", layout="centered")

st.markdown(
    "<h2 style='text-align: center; color: #00bcd4;'>👁️ VisionAid"
    " Assistant</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center;'>Real-time Detection & Voice Alerts</p>",
    unsafe_allow_html=True,
)

# 1. تحميل النموذج
@st.cache_resource
def load_model():
  return YOLO("yolo26n.pt")


model = load_model()

# الفئات الذكية المطلوبة
SMART_CLASSES = [0, 2, 3, 9, 56, 57, 59, 60, 61, 62, 68, 69, 71, 72]

# المتغير المشترك لتخزين رسالة التنبيه اللحظية
if "alert_msg" not in st.session_state:
  st.session_state.alert_msg = "Path is clear"


# 2. معالج الفيديو
class VideoProcessor(VideoTransformerBase):

  def __init__(self):
    self.latest_alert = "Path is clear"

  def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape
    total_area = h * w

    results = model.predict(
        source=img,
        imgsz=320,
        conf=0.35,
        classes=SMART_CLASSES,
        verbose=False,
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
          self.latest_alert = f"{closest_name} ahead"
      else:
        self.latest_alert = "Path is clear"
    else:
      self.latest_alert = "Path is clear"

    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")


# 3. إعدادات البث بالكاميرا الخلفية
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

ctx = webrtc_streamer(
    key="visionaid",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {"facingMode": "environment"},
        "audio": False,
    },
    async_processing=True,
)

# 4. عنصر نطق الصوت مع زر تفاعلي لتجاوز قيود المتصفح
st.markdown("---")
st.markdown("### 🔊 Voice Navigation Control")

# نمرر كود تفاعلي بالكامل داخل المتصفح لنطق الصوت
st.components.v1.html(
    """
<!DOCTYPE html>
<html>
<head>
<style>
  .voice-btn {
    background-color: #00bcd4;
    color: white;
    border: none;
    padding: 12px 20px;
    font-size: 16px;
    font-weight: bold;
    border-radius: 8px;
    cursor: pointer;
    width: 100%;
    margin-bottom: 10px;
  }
  .voice-btn.active {
    background-color: #4caf50;
  }
  #display-text {
    font-size: 15px;
    color: #333;
    font-family: sans-serif;
    text-align: center;
  }
</style>
</head>
<body>

<button id="activate-btn" class="voice-btn" onclick="enableVoice()">🔊 Tap to Enable Audio Alerts</button>
<div id="display-text">Audio is standby. Tap button above.</div>

<script>
  let synth = window.speechSynthesis;
  let isVoiceEnabled = false;
  let lastSpoken = "";
  let lastTime = 0;
  let voiceInstance = null;

  function loadVoice() {
    let voices = synth.getVoices();
    voiceInstance = voices.find(v => (v.name.includes("Google") || v.name.includes("Samantha") || v.name.includes("Natural")) && v.lang.startsWith("en")) 
                    || voices.find(v => v.lang.startsWith("en"));
  }
  loadVoice();
  if (speechSynthesis.onvoiceschanged !== undefined) {
    speechSynthesis.onvoiceschanged = loadVoice;
  }

  function enableVoice() {
    isVoiceEnabled = true;
    let btn = document.getElementById("activate-btn");
    btn.textContent = "✅ Voice Alerts Active";
    btn.className = "voice-btn active";
    document.getElementById("display-text").textContent = "Listening for obstacles...";
    
    // تشغيل جملة ترحيبية فورية لتأكيد منح الصلاحية
    let utter = new SpeechSynthesisUtterance("VisionAid audio is now ready");
    if (voiceInstance) utter.voice = voiceInstance;
    synth.speak(utter);
  }

  function speakText(text) {
    if (!isVoiceEnabled || !text || text.includes("clear") || synth.speaking) return;
    let now = Date.now();
    if (now - lastTime < 2500) return; // مهلة التكرار لمنع الإزعاج

    let utter = new SpeechSynthesisUtterance(text);
    if (voiceInstance) utter.voice = voiceInstance;
    utter.rate = 1.05;
    utter.pitch = 1.1;
    synth.speak(utter);
    lastTime = now;
  }

  // فحص دوري لنص التحذيرات من واجهة العرض ونطقها
  setInterval(() => {
    try {
      const elements = window.parent.document.querySelectorAll("div, p, span");
      for (let el of elements) {
        let txt = el.innerText || "";
        if (txt.includes("ahead") || txt.includes("detected")) {
          if (txt !== lastSpoken) {
            lastSpoken = txt;
            document.getElementById("display-text").textContent = "Alert: " + txt;
            speakText(txt);
          }
          break;
        }
      }
    } catch (e) {}
  }, 600);
</script>

</body>
</html>
""",
    height=130,
)
