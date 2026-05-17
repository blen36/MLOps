import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR = 'Rock-Paper-Scissors/train'
TEST_DIR = 'Rock-Paper-Scissors/test'

# rescale=1./255 переводит пиксели из диапазона 0-255 в диапазон 0-1
# Аугментация создаёт слегка изменённые версии изображений
# Это помогает уменьшить переобучение, потому что модель не просто запоминает картинки
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

# Для validation аугментацию не используем
# Проверочные данные должны оставаться обычными, чтобы честно оценивать модель
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(64, 64),      # все картинки приводятся к размеру 64x64
    batch_size=32,             # модель обучается пачками по 32 изображения
    class_mode='categorical',  # нужно для нескольких классов: paper, rock, scissors
    subset='training',
    color_mode='rgb'           # даже если PNG, изображение будет читаться как RGB
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
    shuffle=False              # для теста лучше не перемешивать данные
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),
    # Первый сверточный слой
    # 32 — количество фильтров, то есть сколько типов признаков ищет слой
    # (3,3) — размер фильтра
    # входное изображение 64x64, 3 цветовых канала RGB
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Dropout случайно отключает часть нейронов во время обучения
    # Это помогает бороться с переобучением
    tf.keras.layers.Dropout(0.25),
    # Второй сверточный слой ищет уже более сложные признаки
    # Например: форму пальцев, контур ладони, силуэт жеста
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # Снова уменьшаем размер признаков
    tf.keras.layers.MaxPooling2D(2, 2),
    # Dropout уменьшает вероятность того, что модель просто запомнит train-картинки
    tf.keras.layers.Dropout(0.25),
    # Третий сверточный слой добавлен для поиска ещё более сложных признаков
    # Например: общая форма жеста и расположение пальцев
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    # Снова уменьшаем размер признаков
    tf.keras.layers.MaxPooling2D(2, 2),
    # Dropout после сверточных слоев помогает уменьшить переобучение
    tf.keras.layers.Dropout(0.25),
    # Flatten превращает многомерные признаки в один длинный список чисел
    tf.keras.layers.Flatten(),
    # Полносвязный слой анализирует найденные признаки
    # 128 — количество нейронов
    tf.keras.layers.Dense(128, activation='relu'),
    # Dropout перед выходным слоем помогает модели лучше обобщать данные
    tf.keras.layers.Dropout(0.5),
    # Выходной слой
    # Количество нейронов равно количеству классов: paper, rock, scissors
    # softmax выдаёт вероятность для каждого класса
    tf.keras.layers.Dense(len(train_data.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=10)

loss, accuracy = model.evaluate(val_data)
print(f'Точность модели на validation: {accuracy * 100:.2f}%')

test_loss, test_accuracy = model.evaluate(test_data)
print(f'Точность модели на test: {test_accuracy * 100:.2f}%')

# Выводим, какие классы нашла модель
# Например: {'paper': 0, 'rock': 1, 'scissors': 2}
print('Классы модели:', train_data.class_indices)

img = image.load_img('5211023283132767925.jpg', target_size=(64, 64), color_mode='rgb')
img_array = image.img_to_array(img) / 255.0
# expand_dims добавляет размерность batch
# Было: 64x64x3
# Стало: 1x64x64x3
# Модель всегда ожидает пачку изображений, даже если картинка одна
img_array = np.expand_dims(img_array, axis=0)
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