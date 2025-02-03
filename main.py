import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Input
from sklearn.model_selection import train_test_split
import albumentations as A
import cv2
import pandas as pd
import random
import matplotlib.pyplot as plt

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Dataset directory
DATADIR = 'track'
COLUMNS = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(DATADIR, 'driving_log.csv'), names=COLUMNS)

# Image Augmentation
AUGMENTATIONS = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2),
    A.Rotate(limit=10, p=0.2),
    A.Blur(blur_limit=3, p=0.1)
])

# Load and preprocess images
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (200, 66))
    return image

# Augment Image
def augment_image(image):
    return AUGMENTATIONS(image=image)['image']

# Load dataset images & labels
def load_img_steering(datadir, df):
    img_paths, steerings, throttles, reverses, speeds = [], [], [], [], []
    for i in range(len(df)):
        row = df.iloc[i]
        img_paths.append(os.path.join(datadir, row['center'].strip()))
        steerings.append(float(row['steering']))
        throttles.append(float(row['throttle']))
        reverses.append(float(row['reverse']))
        speeds.append(float(row['speed']))
    return np.array(img_paths), np.array(steerings), np.array(throttles), np.array(reverses), np.array(speeds)

# Data split
image_paths, steerings, throttles, reverses, speeds = load_img_steering(DATADIR + '/IMG', data)
X_train, X_valid, y_train, y_valid, t_train, t_valid, r_train, r_valid, s_train, s_valid = train_test_split(
    image_paths, steerings, throttles, reverses, speeds, test_size=0.2, random_state=42)

print(image_paths[0])
# Preprocessing Function
def preprocess_image(img):
    img = img[60:135, :, :]  # Crop unnecessary parts
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert to YUV
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Reduce noise
    img = cv2.resize(img, (200, 66))  # Resize to NVIDIA Input Size
    img = img / 255.0  # Normalize
    return img

# Data Generator
def data_generator(image_paths, steering, throttle, reverse, speed, batch_size, training=True):
    def generator():
        while True:
            batch_img, batch_steering, batch_throttle, batch_reverse, batch_speed = [], [], [], [], []
            for _ in range(batch_size):
                idx = random.randint(0, len(image_paths) - 1)
                img = load_image(image_paths[idx])
                img = augment_image(img) if training else img
                img = preprocess_image(img)

                batch_img.append(img)
                batch_steering.append([steering[idx]])  # Ensure correct shape
                batch_throttle.append([throttle[idx]])
                batch_reverse.append([reverse[idx]])
                batch_speed.append([speed[idx]])

            yield (np.array(batch_img, dtype=np.float32),
                   {'steering_out': np.array(batch_steering, dtype=np.float32),
                    'throttle_out': np.array(batch_throttle, dtype=np.float32),
                    'reverse_out': np.array(batch_reverse, dtype=np.float32),
                    'speed_out': np.array(batch_speed, dtype=np.float32)})

    output_signature = (
        tf.TensorSpec(shape=(None, 66, 200, 3), dtype=tf.float32),
        {'steering_out': tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
         'throttle_out': tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
         'reverse_out': tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
         'speed_out': tf.TensorSpec(shape=(None, 1), dtype=tf.float32)}
    )

    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)

# Define Multi-Output NVIDIA Model
def nvidia_model():
    inputs = Input(shape=(66, 200, 3))

    x = Conv2D(24, (5, 5), strides=(2, 2), activation='elu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(36, (5, 5), strides=(2, 2), activation='elu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(48, (5, 5), strides=(2, 2), activation='elu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(100, activation='elu')(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation='elu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='elu')(x)
    x = Dropout(0.5)(x)

    # Separate outputs
    steering_out = Dense(1, name='steering_out')(x)
    throttle_out = Dense(1, name='throttle_out')(x)
    reverse_out = Dense(1, name='reverse_out')(x)
    speed_out = Dense(1, name='speed_out')(x)

    model = Model(inputs=inputs, outputs=[steering_out, throttle_out, reverse_out, speed_out])

    model.compile(loss={'steering_out': 'mse',
                        'throttle_out': 'mse',
                        'reverse_out': 'mse',
                        'speed_out': 'mse'},
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

    return model

# Initialize Model
model = nvidia_model()

# Learning Rate Scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

# Training
history = model.fit(
    data_generator(X_train, y_train, t_train, r_train, s_train, batch_size=128, training=True),
    steps_per_epoch=300,
    epochs=20,
    validation_data=data_generator(X_valid, y_valid, t_valid, r_valid, s_valid, batch_size=128, training=False),
    validation_steps=100,
    callbacks=[lr_scheduler],
    verbose=1,
    shuffle=True
)

# Save Model
model.save('advanced_self_driving_model.h5')
