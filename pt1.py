import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.utils import class_weight
import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR_TRAIN = "C:\\CancerAI\\Training"
DATA_DIR_TEST = "C:\\CancerAI\\Testing"

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.3),
    tf.keras.layers.RandomContrast(0.2),
])

def load_and_preprocess_data(data_dir):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode="binary"
    )

    def normalize(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def augment_map_function(image, label):
        augmented_image = data_augmentation(image, training=True)
        return augmented_image, label

    dataset = dataset.map(normalize)
    if data_dir == DATA_DIR_TRAIN:
      dataset = dataset.map(augment_map_function)
    return dataset

train_dataset = load_and_preprocess_data(DATA_DIR_TRAIN)
test_dataset = load_and_preprocess_data(DATA_DIR_TEST)

train_labels = np.concatenate([y.numpy() for x, y in train_dataset], axis=0)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels.flatten())
class_weights = {0: class_weights[0], 1: class_weights[1]}

base_model = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 1))
x = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=0, restore_best_weights=True)

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset,
    callbacks=[early_stopping],
    class_weight=class_weights
)

test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

model.save('C:\\CancerAI\\model.keras')

def predict_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE, color_mode="grayscale")
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return "Meningioma Detected" if prediction > 0.6 else "No Tumor", prediction

print(predict_image("C:\\CancerAI\\Testing\\Tumor\\Te-me_0012.jpg"))