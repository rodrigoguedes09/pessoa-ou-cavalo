import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
os.environ['MLFLOW_TRACKING_URI'] = 'mlruns' 
from mlflow_config import mlflow_cfg
import mlflow

# Configurações
CONFIG = {
    "input_shape": (224, 224, 3),
    "batch_size": 32,
    "epochs": 15,
    "learning_rate": 1e-3,
    "class_names": ["horse", "human"],  # Garante ordem consistente
    "paths": {
        "train": os.path.join("data", "splits", "train"),
        "val": os.path.join("data", "splits", "validation"),
        "model": os.path.join("model", "horse_human_mobilenetv2.h5"),
        "label_map": os.path.join("model", "label_map.json")
    }
}

# Pré-processamento com validação de dados
def create_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        CONFIG["paths"]["train"],
        target_size=CONFIG["input_shape"][:2],
        batch_size=CONFIG["batch_size"],
        class_mode='categorical',
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        CONFIG["paths"]["val"],
        target_size=CONFIG["input_shape"][:2],
        batch_size=CONFIG["batch_size"],
        class_mode='categorical',
        shuffle=False
    )

    # Salva o mapeamento de classes para uso futuro
    with open(CONFIG["paths"]["label_map"], 'w') as f:
        json.dump(train_gen.class_indices, f)

    return train_gen, val_gen

# Construção do Modelo
def build_model():
    base_model = MobileNetV2(
        input_shape=CONFIG["input_shape"],
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False  # Congelamento mais explícito

    model = models.Sequential([
        base_model,
        layers.Dropout(0.2),  # Regularização adicional
        layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=CONFIG["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model

# Treinamento com callbacks
def train_model(model, train_gen, val_gen):
    mlflow_cfg.setup()  # Inicializa o MLflow
    
    callbacks = [
        ModelCheckpoint(
            CONFIG["paths"]["model"],
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]

    with mlflow.start_run():
        # Log dos parâmetros
        mlflow.log_params({
            "batch_size": CONFIG["batch_size"],
            "epochs": CONFIG["epochs"],
            "learning_rate": CONFIG["learning_rate"],
            "base_model": "MobileNetV2"
        })
        
        history = model.fit(
            train_gen,
            epochs=CONFIG["epochs"],
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Log adicional manual se necessário
        mlflow.log_artifact(CONFIG["paths"]["label_map"])
        
    return history

if __name__ == "__main__":
    # Garante que os diretórios existam
    os.makedirs(os.path.dirname(CONFIG["paths"]["model"]), exist_ok=True)
    
    train_gen, val_gen = create_generators()
    model = build_model()
    history = train_model(model, train_gen, val_gen)

    print(f"\n Treinamento concluído! Modelo salvo em: {CONFIG['paths']['model']}")