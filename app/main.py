# main.py
import os
from data_pipeline import *

# Настройка
setup_tensorflow()

# Пути
HOME = os.getcwd()
print(HOME)
print('-----------------------')

IMAGE_PATH = "/home/oleg/projects/datasets/construction"
print(f"Содержимое папки: {os.listdir(IMAGE_PATH)}")

# Создание пайплайна
train_data, validation_data, train_data_raw, class_names, num_classes = create_data_pipeline(
    image_path=IMAGE_PATH,
    height=224,
    width=224, 
    batch_size=32
)

# Расчет весов классов
print("🎯 Расчет весов классов из тренировочных данных...")
class_weights = get_class_weights_from_dataset(train_data_raw, class_names)

# Просмотр аугментированных изображений
show_augmented_batch(train_data, class_names)

print("✅ Все готово для обучения модели!")
print(f"Количество классов: {num_classes}")
print(f"Названия классов: {class_names}")
print(f"Веса классов: {class_weights}")