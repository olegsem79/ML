# optuna_optimizer.py
import optuna
from optuna.integration import TFKerasPruningCallback
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetV2S, EfficientNetV2M, EfficientNetV2L
from tensorflow.keras import Model
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, Dropout, 
                                   BatchNormalization, Activation)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import datetime
from tensorflow.keras.callbacks import TensorBoard

# 🔧 СЛОВАРЬ MODELS:
MODELS = {
    'EfficientNetV2S': EfficientNetV2S,
    'EfficientNetV2M': EfficientNetV2M, 
    'EfficientNetV2L': EfficientNetV2L,
    'ResNet50': ResNet50,
}

def setup_environment():
    """Настройка окружения для Optuna"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
    tf.get_logger().setLevel('ERROR')

def residual_block(x, units, dropout_rate=0.5, l2_reg=1e-5, activation='relu'):
    """Оптимизированный residual block для небольших датасетов"""
    x = Dense(units, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(dropout_rate)(x)
    return x

def create_optuna_model(trial, num_classes, model_name='EfficientNetV2S', height=224, width=224):
    """Создает модель с параметрами которые предлагает Optuna"""
    
    # Параметры от Optuna
    units_2 = trial.suggest_categorical('units_2', [128, 256]) 
    units_3 = trial.suggest_categorical('units_3', [64, 128])
    dropout_2 = trial.suggest_categorical('dropout_2', [0.4, 0.5, 0.6])
    dropout_3 = trial.suggest_categorical('dropout_3', [0.2, 0.3, 0.4])
    l2_reg = trial.suggest_categorical('l2_reg', [1e-5, 1e-4, 1e-3])
    learning_rate = trial.suggest_categorical('learning_rate', [1e-5, 1e-4, 1e-3])
    activation = trial.suggest_categorical('activation', ['relu', 'elu', 'selu'])

    # Базовая модель
    inputs = tf.keras.Input(shape=(height, width, 3))
    base_model = MODELS[model_name](
        weights='imagenet',
        include_top=False,
        input_shape=(height, width, 3)
    )
    base_model.trainable = False
    x = base_model(inputs, training=False)
    
    # Классификатор
    x = GlobalAveragePooling2D()(x)
    x = residual_block(x, units_2, dropout_2, l2_reg, activation) 
    x = residual_block(x, units_3, dropout_3, l2_reg, activation)
    
    # Выходной слой
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Компиляция
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def run_optuna_optimization(
    train_data,
    validation_data, 
    num_classes,
    model_name='EfficientNetV2S',
    n_trials=20,
    height=224,
    width=224
):
    """Запускает оптимизацию Hyperparameters с помощью Optuna"""
    
    setup_environment()
    
    def objective(trial):
        """Целевая функция для Optuna"""
        model = create_optuna_model(trial, num_classes, model_name, height, width)
        
        history = model.fit(
            train_data,
            epochs=15,
            validation_data=validation_data,
            verbose=1,
            callbacks=[
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                TFKerasPruningCallback(trial, 'val_accuracy')
            ]
        )
        
        best_val_accuracy = max(history.history['val_accuracy'])
        return best_val_accuracy

    print("🚀 ЗАПУСК OPTUNA OPTIMIZATION...")
    
    study = optuna.create_study(
        direction='maximize',
        study_name=f'{model_name}_OPTUNA',
        storage='sqlite:///my_optuna_study.db',
        load_if_exists=True
    )

    study.optimize(objective, n_trials=n_trials)

    print("\n🎯 OPTUNA ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
    print(f"🏆 ЛУЧШИЕ ПАРАМЕТРЫ:")
    print(f"  Accuracy: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    return study

def create_final_model(study, num_classes, model_name='EfficientNetV2S', height=224, width=224):
    """Создает финальную модель с лучшими параметрами"""
    print("\n🔧 СОЗДАЕМ ФИНАЛЬНУЮ НЕ ОБУЧЕННУЮ МОДЕЛЬ...")
    best_model = create_optuna_model(study.best_trial, num_classes, model_name, height, width)
    best_model.summary()
    
    # Сохраняем модель с лучшими параметрами
    best_model.save('model_optuna_best_params.keras')
    print("💾 Модель сохранена как 'model_optuna_best_params.keras'")
    
    return best_model

def visualize_optuna_results(study):
    """Визуализирует результаты Optuna"""
    print("\n📈 ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ...")
    
    try:
        # 1. График оптимизации
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.show()

        # 2. Важность параметров
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.show()

        # 3. Параллельные координаты
        fig3 = optuna.visualization.plot_parallel_coordinate(study)
        fig3.show()

        # 4. Распределение параметров
        fig4 = optuna.visualization.plot_contour(study)
        fig4.show()

        # 5. Сравнение распределений
        fig5 = optuna.visualization.plot_slice(study)
        fig5.show()

        # 6. Timeline оптимизации
        fig6 = optuna.visualization.plot_timeline(study)
        fig6.show()
        
    except Exception as e:
        print(f"⚠️ Ошибка визуализации: {e}")

def train_final_model(
    model,
    train_data,
    validation_data,
    class_weights=None,
    epochs=70,
    model_name='best_model'
):
    """Обучает финальную модель с лучшими параметрами"""
    
    # Callbacks
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq='epoch',
        profile_batch=0
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', 
            patience=15, 
            restore_best_weights=True, 
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'{model_name}_callback.keras', 
            monitor='val_accuracy', 
            save_best_only=True, 
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            patience=7, 
            factor=0.3, 
            verbose=1
        ),
        tensorboard_callback
    ]
    
    print("🚀 ОБУЧАЕМ ФИНАЛЬНУЮ МОДЕЛЬ...")
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, validation_data, model_name="model"):
    """Оценивает модель и сохраняет результаты"""
    print(f"\n📊 ОЦЕНКА {model_name.upper()}...")
    val_results = model.evaluate(validation_data, verbose=1)
    val_loss, val_accuracy = val_results[0], val_results[1]
    
    print(f"✅ Результаты {model_name}:")
    print(f"   - Validation Accuracy: {val_accuracy:.4f}")
    print(f"   - Validation Loss: {val_loss:.4f}")
    
    # Сохраняем модель с accuracy в имени
    model.save(f'{model_name}_{val_accuracy:.4f}.keras')
    print(f"💾 Модель сохранена как '{model_name}_{val_accuracy:.4f}.keras'")
    
    return val_loss, val_accuracy