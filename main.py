import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

# Загрузка предобученной модели
model = load_model('digit_model.h5')

from scipy.ndimage.measurements import center_of_mass
import math
from PIL import Image, ImageOps
import numpy as np

def getBestShift(img):
    cy, cx = center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    return shiftx, shifty

def shift(img, sx, sy, rows, cols):
    # Создаем новое изображение с тем же размером
    shifted = np.zeros_like(img)
    
    # Вычисляем новые координаты после сдвига по оси X
    if sx > 0:
        shifted[:, sx:] = img[:, :-sx]
    elif sx < 0:
        shifted[:, :sx] = img[:, -sx:]
    else:
        shifted = img
    
    # Вычисляем новые координаты после сдвига по оси Y
    if sy > 0:
        shifted[sy:, :] = shifted[:rows-sy, :]
    elif sy < 0:
        shifted[:sy, :] = shifted[-sy:, :]
    
    return shifted

def rec_digit(uploaded_file):
    # Загрузка изображения
    img = Image.open(uploaded_file).convert('L')  # Конвертация в градации серого
    img = ImageOps.invert(img)  # Инверсия цветов (белые цифры на чёрном фоне)
    gray = np.array(img)  # Преобразование в массив NumPy

    # Применяем пороговую обработку
    threshold = 128
    gray = np.where(gray > threshold, 255, 0).astype(np.uint8)

    # Удаляем нулевые строки и столбцы
    while np.sum(gray[0]) == 0:
        gray = gray[1:]
    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, axis=1)
    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]
    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, axis=1)

    rows, cols = gray.shape

    # Изменяем размер, чтобы помещалось в box 20x20 пикселей
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))

    resized_img = Image.fromarray(gray).resize((cols, rows), Image.Resampling.LANCZOS)
    gray = np.array(resized_img)

    # Расширяем до размера 28x28
    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.pad(gray, (rowsPadding, colsPadding), 'constant', constant_values=0)

    # Сдвигаем центр масс
    shiftx, shifty = getBestShift(gray)
    gray = shift(gray, shiftx, shifty, rows, cols)

    # Сохраняем обработанное изображение
    processed_img = Image.fromarray(gray.astype(np.uint8))
    output_path = f"gray_{uploaded_file.name}"  # Используем имя загруженного файла
    processed_img.save(output_path)

    # Нормализация и подготовка данных
    img = gray / 255.0
    img = np.array(img).reshape(-1, 28, 28, 1)
    out = str(np.argmax(model.predict(img)))
    return out

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
st.markdown('<h1 style="color: white;">Распознавание AZOV рукописных цифр</h1>', unsafe_allow_html=True)

# Загрузка изображения
uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Открытие и отображение изображения
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение.', use_container_width=True)
    result = rec_digit(uploaded_file)
    # Классификация изображения
    try:
        #result, conf = classify_image(image)
        st.markdown(f'<p style="color: white;">Предсказанная цифра: {result}</p>', unsafe_allow_html=True)
        #st.markdown(f'<p style="color: white;">Уверенность в предсказании: {conf}</p>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Произошла ошибка: {e}")



