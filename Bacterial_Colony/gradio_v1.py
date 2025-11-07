# cd /home/oleg/projects/ML/Bacterial_Colony
# python /home/oleg/projects/ML/Bacterial_Colony/gradio.py

import gradio as gr
from ultralytics import YOLO
import time

# Загружаем модель
model = YOLO('/home/oleg/projects/ML/Bacterial_Colony/runs/classify/train/weights/best.pt')

def classify_bacteria(image):
    """Классификация бактерий"""
    try:
        # Добавляем задержку для красивого spinner
        time.sleep(0.5)
        
        # Предсказание
        results = model(image)
        result = results[0]
        
        if hasattr(result, 'probs') and result.probs is not None:
            probs = result.probs
            top1_idx = probs.top1
            confidence = probs.top1conf.item()
            class_name = model.names[top1_idx]  # ⬅️ ИСПРАВЛЕНО: probs.top1
            
            # Форматируем результат
            result_text = f"🔬 **Результат:** {class_name}\n"
            result_text += f"🎯 **Точность:** {confidence:.2%}\n\n"
            
            # Добавляем топ-3 предсказания
            result_text += "**Топ-3 предсказания:**\n"
            top3_indices = probs.top5[:3]
            for i, idx in enumerate(top3_indices, 1):
                name = model.names[idx]
                conf = probs.data[idx].item()
                result_text += f"{i}. {name}: {conf:.2%}\n"
            
            return result_text
        else:
            return "❌ Не удалось получить вероятности классификации"
            
    except Exception as e:
        return f"❌ Ошибка: {str(e)}"

# Создаем интерфейс
demo = gr.Interface(
    fn=classify_bacteria,
    inputs=gr.Image(type="pil", label="📷 Загрузите изображение бактерий"),
    outputs=gr.Textbox(label="🎯 Результат классификации", lines=8),
    title="🔬 Классификатор бактерий",
    description="Загрузите изображение бактерий для определения вида. Модель покажет топ-3 наиболее вероятных варианта.",
    examples=[
        ["sample1.jpg"],  # Добавьте примеры изображений
        ["sample2.jpg"]
    ],
    allow_flagging="never"
)

# Запускаем
if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )