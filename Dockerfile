# اختيار نسخة بايثون
FROM python:3.11-slim

# تنصيب مكتبات النظام المطلوبة للـ OpenCV و EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- إعداد الصلاحيات الخاصة بـ Hugging Face ---
# المنصة تتطلب تشغيل الكود كمستخدم عادي (ID 1000) لتجنب أخطاء الصلاحيات
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# تحديد مسار العمل
WORKDIR $HOME/app

# نسخ ملف المكتبات وتسطيبها (مع نقل الملكية للمستخدم)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- التحميل المسبق لموديلات الذكاء الاصطناعي الضخمة ---
# هذه الخطوة ضرورية جداً لكي تكون الموديلات جاهزة داخل السيرفر
RUN python -c "import cv2, numpy as np; from deepface import DeepFace; img = np.zeros((10,10,3), np.uint8); cv2.imwrite('dummy.jpg', img); DeepFace.represent('dummy.jpg', model_name='Facenet512', detector_backend='retinaface', enforce_detection=False)"
RUN python -c "import easyocr; easyocr.Reader(['ar', 'en'], gpu=False)"

# نسخ باقي الكود
COPY --chown=user . .

# أمر تشغيل السيرفر المتوافق مع Hugging Face Spaces (البورت الافتراضي 7860)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]