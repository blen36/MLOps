import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR = 'Rock-Paper-Scissors/train'
TEST_DIR = 'Rock-Paper-Scissors/test'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(64, 64),
    batch_size=32,             # по 32 изобр за раз
    class_mode='categorical',
    subset='training',
    color_mode='rgb'
)

val_data = val_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    color_mode='rgb'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),  #полносвязн
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(len(train_data.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=7)

loss, accuracy = model.evaluate(val_data)
print(f'Точность модели на validation: {accuracy * 100:.2f}%')

test_loss, test_accuracy = model.evaluate(test_data)
print(f'Точность модели на test: {test_accuracy * 100:.2f}%')

print('Классы модели:', train_data.class_indices)

img = image.load_img('5237790176451438131.jpg', target_size=(64, 64), color_mode='rgb')
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Было: 64x64x3  Стало: 1x64x64x3
prediction = model.predict(img_array)
class_names = list(train_data.class_indices.keys())
predicted_index = np.argmax(prediction[0])
predicted_class = class_names[predicted_index]
confidence = prediction[0][predicted_index]
print(f'Предсказанный класс: {predicted_class}')
print(f'Уверенность модели: {confidence * 100:.2f}%')
print('Вероятности по классам:')
for class_name, probability in zip(class_names, prediction[0]):
    print(f'{class_name}: {probability * 100:.2f}%')