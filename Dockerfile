# اختيار نسخة بايثون
FROM python:3.11-slim

# تنصيب مكتبات النظام المطلوبة للـ OpenCV و EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# تحديد مسار العمل
WORKDIR /app

# نسخ ملف المكتبات وتسطيبها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- التحميل المسبق لموديلات الذكاء الاصطناعي الضخمة ---
# هذه الخطوة ضرورية جداً لـ Render لكي تكون الموديلات جاهزة داخل السيرفر
# ولا يضطر لتحميلها (مئات الميجابايت) عند وصول أول طلب مما قد يسبب انقطاعاً.

# 1. تحميل موديلات DeepFace (Facenet512 & retinaface)
RUN python -c "import cv2, numpy as np; from deepface import DeepFace; img = np.zeros((10,10,3), np.uint8); cv2.imwrite('dummy.jpg', img); DeepFace.represent('dummy.jpg', model_name='Facenet512', detector_backend='retinaface', enforce_detection=False)"

# 2. تحميل موديلات EasyOCR (عربي وإنجليزي)
RUN python -c "import easyocr; easyocr.Reader(['ar', 'en'], gpu=False)"

# نسخ باقي الكود
COPY . .

# أمر تشغيل السيرفر المتوافق مع Render
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"
