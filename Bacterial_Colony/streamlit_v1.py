# streamlit run streamlit_v1.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import time

# Настройка страницы
st.set_page_config(
    page_title="🔬 Классификатор бактерий",
    page_icon="🦠",
    layout="wide"
)

# Заголовок
st.title("🔬 Классификатор бактерий")
st.markdown("Загрузите изображение бактерий для определения вида с помощью YOLO модели")

# Загрузка модели с кэшированием
@st.cache_resource
def load_model():
    """Загружаем YOLO модель один раз"""
    try:
        model = YOLO('/home/oleg/projects/ML/Bacterial_Colony/runs/classify/train/weights/best.pt')
        st.success("✅ Модель загружена успешно!")
        st.info(f"🎯 Доступно классов: {len(model.names)}")
        return model
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None

# Загружаем модель
model = load_model()

# Боковая панель с информацией
with st.sidebar:
    st.header("ℹ️ Информация")
    st.markdown("""
    ### Как использовать:
    1. Загрузите изображение бактерий
    2. Нажмите кнопку 'Определить'
    3. Получите результат классификации
    
    ### Поддерживаемые форматы:
    - JPG, JPEG, PNG, TIFF
    - Рекомендуемое разрешение: 640x640px
    """)
    
    if model and hasattr(model, 'names'):
        st.subheader("📊 Доступные классы")
        classes_df = pd.DataFrame(list(model.names.values()), columns=["Классы бактерий"])
        st.dataframe(classes_df, height=300, use_container_width=True)

# Основная область
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Загрузка изображения")
    
    # Загрузка файла
    uploaded_file = st.file_uploader(
        "Выберите изображение...", 
        type=['jpg', 'jpeg', 'png', 'tiff'],
        help="Загрузите изображение бактерий для классификации"
    )
    
    if uploaded_file is not None:
        # Показываем изображение
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", use_column_width=True)
        
        # Кнопка для классификации
        if st.button("🎯 Определить бактерию", type="primary", use_container_width=True):
            if model is None:
                st.error("Модель не загружена!")
            else:
                with st.spinner("🔍 Анализируем изображение..."):
                    # Имитация загрузки для красоты
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Классификация
                    try:
                        results = model(image)
                        result = results[0]
                        
                        if hasattr(result, 'probs') and result.probs is not None:
                            probs = result.probs
                            
                            # Основной результат
                            top1_idx = probs.top1
                            confidence = probs.top1conf.item()
                            class_name = model.names[top1_idx]
                            
                            # Топ-5 предсказаний
                            top5_indices = probs.top5[:5]
                            top5_data = []
                            
                            for idx in top5_indices:
                                name = model.names[idx]
                                conf = probs.data[idx].item()
                                top5_data.append({
                                    "Класс": name,
                                    "Уверенность": f"{conf:.2%}"
                                })
                            
                            # Выводим результаты в col2
                            with col2:
                                st.subheader("📊 Результаты")
                                
                                # Основной результат
                                st.success(f"**🎯 Основной результат:** {class_name}")
                                st.metric(
                                    label="Уверенность модели", 
                                    value=f"{confidence:.2%}"
                                )
                                
                                # Топ-5 предсказаний
                                st.subheader("🏆 Топ-5 предсказаний")
                                top5_df = pd.DataFrame(top5_data)
                                st.dataframe(top5_df, use_container_width=True)
                                
                                # Визуализация уверенности
                                st.subheader("📈 Визуализация уверенности")
                                
                                # Создаем bar chart
                                chart_data = {
                                    "Классы": [model.names[idx] for idx in top5_indices],
                                    "Уверенность": [probs.data[idx].item() for idx in top5_indices]
                                }
                                st.bar_chart(
                                    pd.DataFrame(chart_data).set_index("Классы"),
                                    use_container_width=True
                                )
                        
                        else:
                            st.error("❌ Не удалось получить вероятности классификации")
                            
                    except Exception as e:
                        st.error(f"❌ Ошибка при классификации: {str(e)}")

with col2:
    if uploaded_file is None:
        st.info("👆 Загрузите изображение слева для начала классификации")
        
        # Показываем пример интерфейса
        st.subheader("📋 Пример вывода:")
        st.success("**🎯 Основной результат:** Escherichia.coli")
        st.metric(label="Уверенность модели", value="94.23%")
        
        example_data = {
            "Класс": ["Escherichia.coli", "Pseudomonas.aeruginosa", "Staphylococcus.aureus"],
            "Уверенность": ["94.23%", "3.45%", "1.12%"]
        }
        st.dataframe(pd.DataFrame(example_data), use_container_width=True)

# Футер
st.markdown("---")
st.markdown("### 🦠 Классификатор бактерий • Powered by YOLO • Streamlit")