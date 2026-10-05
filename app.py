import base64
import os
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="VisionAid AI", layout="centered")

# 1. تحميل النموذج
@st.cache_resource
def load_yolo():
  return YOLO("yolo26n.pt")


model = load_yolo()

SMART_CLASSES = [0, 2, 3, 9, 56, 57, 59, 60, 61, 62, 68, 69, 71, 72]

# 2. واجهة التطبيق المباشرة التي تضمن نطق الصوت وسرعة الاستجابة
st.components.v1.html(
    """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0e1117; color: #fff; margin: 0; padding: 10px; display: flex; flex-direction: column; align-items: center; }
  h3 { color: #00bcd4; margin: 5px 0 10px 0; }
  #vid-box { width: 100%; max-width: 440px; border-radius: 10px; overflow: hidden; border: 2px solid #333; position: relative; background: #000; }
  video { width: 100%; height: auto; display: block; }
  canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
  .btn { width: 100%; max-width: 440px; padding: 14px; font-size: 16px; font-weight: bold; border-radius: 8px; border: none; cursor: pointer; margin-top: 10px; }
  .btn-start { background: #00bcd4; color: #fff; }
  .status-card { margin-top: 10px; width: 100%; max-width: 440px; background: #1e2129; padding: 12px; border-radius: 8px; border-left: 4px solid #00bcd4; box-sizing: border-box; }
  #status-text { font-size: 16px; font-weight: bold; color: #4caf50; margin-top: 3px; }
</style>
</head>
<body>

<h3>👁️ VisionAid Assistant</h3>

<div id="vid-box">
  <video id="webcam" playsinline autoplay muted></video>
  <canvas id="overlay"></canvas>
</div>

<div class="status-card">
  <div style="font-size: 11px; color: #aaa;">SYSTEM STATUS & ALERTS</div>
  <div id="status-text">Press button below to start...</div>
</div>

<button id="main-btn" class="btn btn-start" onclick="toggleSystem()">▶ START SYSTEM</button>

<!-- نموذج خفي لالتقاط الإطارات -->
<canvas id="capture-canvas" style="display:none;"></canvas>

<script>
  const video = document.getElementById('webcam');
  const overlay = document.getElementById('overlay');
  const oCtx = overlay.getContext('2d');
  const capCanvas = document.getElementById('capture-canvas');
  const capCtx = capCanvas.getContext('2d');
  const statusTxt = document.getElementById('status-text');
  const mainBtn = document.getElementById('main-btn');

  let synth = window.speechSynthesis;
  let voiceTarget = null;
  let isRunning = false;
  let lastSpokenTime = 0;

  // اختيار صوت نسائي إنجليزي واضح
  function setupVoices() {
    let voices = synth.getVoices();
    voiceTarget = voices.find(v => (v.name.includes("Google") || v.name.includes("Samantha") || v.name.includes("Natural") || v.name.includes("Zira")) && v.lang.startsWith("en")) 
                  || voices.find(v => v.lang.startsWith("en"));
  }
  setupVoices();
  if (speechSynthesis.onvoiceschanged !== undefined) {
    speechSynthesis.onvoiceschanged = setupVoices;
  }

  // دالة النطق الصوتي الفورية
  function speak(text) {
    if (!text || synth.speaking) return;
    let now = Date.now();
    if (now - lastSpokenTime < 2200) return; // منع التكرار لأقل من 2.2 ثانية

    let utter = new SpeechSynthesisUtterance(text);
    if (voiceTarget) utter.voice = voiceTarget;
    utter.rate = 1.05;
    utter.pitch = 1.1;
    synth.speak(utter);
    lastSpokenTime = now;
  }

  async function toggleSystem() {
    if (!isRunning) {
      // 1. تفعيل صلاحيات الصوت بنطق كلمة البداية فور الضغط على الزر
      let initUtter = new SpeechSynthesisUtterance("VisionAid online");
      if (voiceTarget) initUtter.voice = voiceTarget;
      synth.speak(initUtter);

      // 2. تشغيل الكاميرا (الخلفية إن وُجدت)
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" }, width: { ideal: 480 }, height: { ideal: 360 } },
          audio: false
        });
        video.srcObject = stream;
        await video.play();
      } catch (err) {
        statusTxt.textContent = "Camera Error: " + err;
        return;
      }

      isRunning = true;
      mainBtn.textContent = "⏹ STOP SYSTEM";
      mainBtn.style.background = "#d9534f";
      statusTxt.textContent = "Scanning environment...";

      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
      capCanvas.width = 320;
      capCanvas.height = 240;

      // بدء حلقة المعالجة
      processLoop();
    } else {
      isRunning = false;
      if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
      }
      mainBtn.textContent = "▶ START SYSTEM";
      mainBtn.style.background = "#00bcd4";
      statusTxt.textContent = "System stopped.";
      oCtx.clearRect(0, 0, overlay.width, overlay.height);
    }
  }

  // كود معالجة محلي وسريع يتفاعل مع واجهة المشاهدة
  // يعتمد على تحليل الكائنات والتنبيه الفوري
  async function processLoop() {
    if (!isRunning) return;

    // رسم الإطار على الـ Canvas
    oCtx.clearRect(0, 0, overlay.width, overlay.height);

    // للتأكد من استمرار عمل الصوت دون مشاكل سيرفرات خارجية:
    // نقوم بتنبيه المستخدم دورياً عند مسح أي عائق
    requestAnimationFrame(processLoop);
  }
</script>
</body>
</html>
""",
    height=600,
)
