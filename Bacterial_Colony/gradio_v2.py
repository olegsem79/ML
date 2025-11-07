import gradio as gr
from ultralytics import YOLO

# Загружаем модель
model = YOLO('/home/oleg/projects/ML/Bacterial_Colony/runs/classify/train/weights/best.pt')

def classify_bacteria(image):
    """Классификация бактерий"""
    results = model(image)
    result = results[0]
    
    if hasattr(result, 'probs'):
        probs = result.probs
        top1_idx = probs.top1
        confidence = probs.top1conf.item()
        class_name = model.names[probs.top1]
        
        return f"🔬 Результат: {class_name}\n🎯 Точность: {confidence:.2%}"
    else:
        return "❌ Не удалось классифицировать"

# Создаем интерфейс
demo = gr.Interface(
    fn=classify_bacteria,
    inputs=gr.Image(type="pil", label="Загрузите изображение бактерий"),
    outputs=gr.Textbox(label="Результат классификации"),
    title="🔬 Классификатор бактерий",
    description="Загрузите изображение бактерий для определения вида"
)

# Запускаем
demo.launch(share=True)  # share=True дает публичную ссылку