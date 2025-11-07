# streamlit run streamlit_v2.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("🔬 Классификатор бактерий")

# Загрузка модели
model = YOLO('/home/oleg/projects/ML/Bacterial_Colony/runs/classify/train/weights/best.pt')

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите изображение бактерий")

if uploaded_file and st.button("Определить"):
    image = Image.open(uploaded_file)
    results = model(image)
    
    if hasattr(results[0], 'probs'):
        probs = results[0].probs
        class_name = model.names[probs.top1]
        confidence = probs.top1conf.item()
        
        st.success(f"Результат: {class_name}")
        st.info(f"Уверенность: {confidence:.2%}")