import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Transfer Learning Model
def create_image_model():
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":

    print("Creating image classification model...")

    model = create_image_model()

    # Dataset path
    DATASET_PATH = "dataset"

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    print("Training model...")

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=5
    )

    # Create model folder
    if not os.path.exists("model"):
        os.makedirs("model")

    model.save("model/image_model.h5")

    # Save label encoder
    label_encoder = LabelEncoder()
    labels = np.array(['safe', 'predator', 'harassment'])
    label_encoder.fit(labels)

    joblib.dump(label_encoder, "model/image_label_encoder.pkl")

    print("Model trained and saved successfully!")
