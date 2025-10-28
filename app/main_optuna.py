# main_optuna.py
import os
from data_pipeline import setup_tensorflow, create_data_pipeline, get_class_weights_from_dataset
from optuna_optimizer import (run_optuna_optimization, create_final_model, 
                            train_final_model, evaluate_model, visualize_optuna_results)
from visualization import (plot_training_history, safe_confusion_matrix_analysis, 
                         test_random_predictions_modern)

def main():
    # Настройка
    setup_tensorflow()
    
    # Пути
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
    print("🎯 Расчет весов классов...")
    class_weights = get_class_weights_from_dataset(train_data_raw, class_names)
    
    # Оптимизация Optuna
    study = run_optuna_optimization(
        train_data=train_data,
        validation_data=validation_data,
        num_classes=num_classes,
        model_name='EfficientNetV2S',
        n_trials=20
    )
    
    # Визуализация результатов Optuna
    visualize_optuna_results(study)
    
    # Создание финальной модели
    best_model = create_final_model(study, num_classes)
    
    # Обучение финальной модели
    final_history = train_final_model(
        model=best_model,
        train_data=train_data,
        validation_data=validation_data,
        class_weights=class_weights,
        epochs=70
    )
    
    # Оценка модели
    val_loss, val_accuracy = evaluate_model(best_model, validation_data, "trained_optuna_model")
    
    # Визуализация результатов
    plot_training_history(final_history)
    safe_confusion_matrix_analysis(best_model, validation_data, class_names)
    test_random_predictions_modern(best_model, validation_data, class_names)
    
    print("🎉 ВСЕ ЗАДАЧИ ВЫПОЛНЕНЫ!")

if __name__ == "__main__":
    main()
    