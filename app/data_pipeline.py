# data_pipeline.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import os

def setup_tensorflow():
    """Настройка TensorFlow для оптимальной производительности"""
    # Оптимизация TensorFlow
    tf.config.optimizer.set_jit(True)  # Включение XLA-компиляции
    tf.config.threading.set_intra_op_parallelism_threads(8)
    
    # Для GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Подавление warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')

def create_data_pipeline(
    image_path, 
    height=224, 
    width=224, 
    batch_size=32, 
    validation_split=0.2,
    seed=42
):
    """
    Создает полный пайплайн данных для обучения
    
    Args:
        image_path (str): Путь к папке с изображениями
        height (int): Высота изображений
        width (int): Ширина изображений  
        batch_size (int): Размер батча
        validation_split (float): Доля валидации
        seed (int): Seed для воспроизводимости
    
    Returns:
        tuple: (train_data, validation_data, train_data_raw, class_names, num_classes)
    """
    print("🔄 СОЗДАЕМ ПАЙПЛАЙН...")
    
    # Загрузка данных
    train_data_raw = tf.keras.preprocessing.image_dataset_from_directory(
        image_path,
        validation_split=validation_split,
        subset='training',
        seed=seed,
        image_size=(height, width),
        batch_size=batch_size,
        shuffle=True
    )

    validation_data_raw = tf.keras.preprocessing.image_dataset_from_directory(
        image_path,
        validation_split=validation_split,
        subset='validation', 
        seed=seed,
        image_size=(height, width),
        batch_size=batch_size,
        shuffle=True
    )

    # Проверка формата
    images, labels = next(iter(train_data_raw))
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels.numpy()[:10]}")

    class_names = train_data_raw.class_names
    num_classes = len(class_names)
    print(f"🏷️ Классы: {class_names}")
    
    # Аугментация
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
    ])

    def preprocess_and_augment(image, label, training=False):
        """Препроцессинг и аугментация"""
        image = tf.cast(image, tf.float32)
        if training:
            image = data_augmentation(image, training=True)
        return image, label

    # Финальные datasets
    train_data = (
        train_data_raw
        .map(lambda x, y: preprocess_and_augment(x, y, training=True),
             num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    validation_data = (
        validation_data_raw
        .map(lambda x, y: preprocess_and_augment(x, y, training=False), 
             num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    # Проверка препроцессинга
    print("🔍 Проверка препроцессинга:")
    for images, labels in train_data.take(1):
        print(f"Shape: {images.shape}")
        print(f"Data range: {tf.reduce_min(images):.3f} to {tf.reduce_max(images):.3f}")
        print(f"Mean: {tf.reduce_mean(images):.3f}")
        break

    print("✅ Пайплайн создан!")
    
    return train_data, validation_data, train_data_raw, class_names, num_classes

def get_class_weights_from_dataset(dataset, class_names):
    """
    Рассчитывает веса классов из tf.data.Dataset
    
    Args:
        dataset: tf.data.Dataset с метками
        class_names: Список названий классов
    
    Returns:
        dict: Словарь весов классов {class_index: weight}
    """
    # Собираем все метки
    all_labels = []
    for images, labels in dataset:
        all_labels.extend(labels.numpy())
    
    y_train = np.array(all_labels)
    
    # Расчет весов
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    weights_dict = dict(enumerate(class_weights))
    
    # Визуализация
    plt.figure(figsize=(12, 5))
    
    # График 1: Распределение классов
    plt.subplot(1, 2, 1)
    unique, counts = np.unique(y_train, return_counts=True)
    bars = plt.bar(range(len(unique)), counts)
    plt.title('Распределение классов')
    plt.xlabel('Класс')
    plt.ylabel('Количество')
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45)
    
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                str(count), ha='center', va='bottom')
    
    # График 2: Веса классов
    plt.subplot(1, 2, 2)
    bars = plt.bar(weights_dict.keys(), weights_dict.values())
    plt.title('Веса классов')
    plt.xlabel('Класс')
    plt.ylabel('Вес')
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45)
    
    for bar, weight in zip(bars, weights_dict.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{weight:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Вывод информации
    print("📊 РАСПРЕДЕЛЕНИЕ КЛАССОВ:")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        count = counts[i] if i < len(counts) else 0
        weight = weights_dict.get(i, 0)
        print(f"{class_name:15} | {count:4} | вес: {weight:.2f}")
    
    return weights_dict

def show_augmented_batch(train_data, class_names):
    """
    Показывает аугментированные изображения из train_data
    
    Args:
        train_data: tf.data.Dataset с аугментированными данными
        class_names: Список названий классов
    """
    print("🎲 АУГМЕНТИРОВАННЫЕ ИЗОБРАЖЕНИЯ ИЗ TRAIN_DATA:")
    
    # Берем один батч из train_data
    for images, labels in train_data.take(1):
        print(f"Размер батча: {images.shape}")
        print(f"Количество изображений: {len(images)}")
        
        plt.figure(figsize=(15, 10))
        
        # Показываем первые 8 изображений из батча
        for i in range(min(8, len(images))):
            plt.subplot(2, 4, i+1)
            
            # Берем изображение из батча
            image = images[i]
            label = labels[i]
            
            # Денормализуем если нужно (из [0,1] в [0,255])
            if tf.reduce_max(image) <= 1.0:
                image = image * 255
            image = tf.cast(image, tf.uint8)
            
            plt.imshow(image)
            plt.title(f'Class: {class_names[label.numpy()]}')
            plt.axis('off')
        
        plt.suptitle('АУГМЕНТИРОВАННЫЕ ИЗОБРАЖЕНИЯ ИЗ TRAIN_DATA', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Проверяем диапазон данных
        print(f"Диапазон данных в батче: {tf.reduce_min(images):.3f} до {tf.reduce_max(images):.3f}")
        break