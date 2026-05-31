import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

IMG_SIZE = (160, 160)

TRAIN_DIR = 'Rock-Paper-Scissors/train'
TEST_DIR = 'Rock-Paper-Scissors/test'
MY_IMAGE_PATH = 'test_gesture.jpg'

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,   # Аугментация данных
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical',
    subset='training',
    color_mode='rgb'
)

val_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    color_mode='rgb'
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights='imagenet' #модель уже обучена на большом наборе изображений
)

base_model.trainable = False  # Замораживаем базовую модель чтобы ее веса не обновлялись

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(160, 160, 3)),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(), #превращает карты признаков в один вектор как Flatten но лучше для предобученых CNN
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(train_data.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=10)

loss, accuracy = model.evaluate(val_data)
print(f'Точность MobileNetV2 на validation: {accuracy * 100:.2f}%')

test_loss, test_accuracy = model.evaluate(test_data)
print(f'Точность MobileNetV2 на test: {test_accuracy * 100:.2f}%')
print('Классы модели:', train_data.class_indices)

img = image.load_img( '5237790176451438131.jpg', target_size=(160, 160), color_mode='rgb')
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)
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