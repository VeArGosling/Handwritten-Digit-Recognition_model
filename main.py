import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

# Загрузка предобученной модели
model = load_model('Digit_Recognition_model.h5')

# Функция предобработки изображения
def preprocess_image(img):
    # Преобразование изображения в градации серого
    img = img.convert('L')  # Конвертация в черно-белое изображение
    img = np.array(img)  # Преобразование в массив NumPy

    # Изменение размера до 28x28 пикселей
    resized_image = Image.fromarray(img).resize((28, 28), Image.Resampling.LANCZOS)

    # Инверсия цветов (MNIST использует белые цифры на черном фоне)
    inverted_image = 255 - np.array(resized_image)

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

# Настройка фона страницы
import base64  # Для работы с Base64

# Настройка фона страницы
def set_background(image_path):
    """
    Устанавливает фоновое изображение для страницы Streamlit.

    :param image_path: Путь к файлу изображения.
    """
    with open(image_path, "rb") as image_file:
        # Чтение файла в байтах и преобразование в Base64
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Определение MIME-типа изображения на основе расширения файла
        if image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
            mime_type = "image/jpeg"
        elif image_path.endswith(".png"):
            mime_type = "image/png"
        else:
            raise ValueError("Неподдерживаемый формат изображения. Используйте JPG или PNG.")
        
        # Внедрение CSS через st.markdown
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:{mime_type};base64,{encoded_string});
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Установка фонового изображения
set_background("background.jpg")  # Убедитесь, что файл "background.jpg" находится в той же директории

# Заголовок приложения
st.title("Распознавание рукописных цифр")

# Загрузка изображения
uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Открытие и отображение изображения
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение.', use_container_width=True)
    
    # Классификация изображения
    try:
        result = classify_image(image)
        st.write(f"Предсказанная цифра: {result}")
    except Exception as e:
        st.error(f"Произошла ошибка: {e}")
