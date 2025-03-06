import streamlit as st
from PIL import Image
import numpy as np
import cv2
from keras.models import load_model

# Загрузка предобученной модели
model = load_model('Digit_Recognition_model.h5')

# Функция предобработки изображения
def preprocess_image(image):
    # Преобразование изображения в градации серого
    image = image.convert('L')  # Конвертация в черно-белое изображение
    image = np.array(image)  # Преобразование в массив NumPy

    # Изменение размера до 28x28 пикселей (стандартный размер для MNIST)
    resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    # Инверсия цветов (MNIST использует белые цифры на черном фоне)
    inverted_image = 255 - resized_image

    # Нормализация значений пикселей в диапазон [0, 1]
    normalized_image = inverted_image / 255.0

    # Добавление размерности для соответствия входу модели (1, 28, 28)
    reshaped_image = normalized_image.reshape(1, 28, 28)

    return reshaped_image

# Функция классификации изображения
def classify_image(image):
    # Предобработка изображения
    processed_image = preprocess_image(image)

    # Получение предсказания от модели
    predictions = model.predict(processed_image)

    # Определение предсказанной цифры
    predicted_digit = np.argmax(predictions, axis=1)[0]

    return predicted_digit

# Заголовок приложения
st.title("Распознавание рукописных цифр")

# Загрузка изображения
uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Открытие и отображение изображения
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение.', use_column_width=True)
    
    # Классификация изображения
    try:
        result = classify_image(image)
        st.write(f"Предсказанная цифра: {result}")
    except Exception as e:
        st.error(f"Произошла ошибка: {e}")
