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
    shifted = np.zeros_like(img)
    
    # Сдвиг по X
    if sx > 0:
        shifted[:, sx:] = img[:, :-sx]
    elif sx < 0:
        shifted[:, :sx] = img[:, -sx:]
    else:
        shifted = img
    
    # Сдвиг по Y
    if sy > 0:
        shifted[sy:, :] = img[:-sy, :]
    elif sy < 0:
        shifted[:sy, :] = img[-sy:, :]
    else:
        shifted = img
    
    return shifted

def rec_digit(uploaded_file):
    # Загрузка изображения
    img = Image.open(uploaded_file).convert('L')
    img = ImageOps.invert(img)
    gray = np.array(img)

    # Пороговая обработка
    threshold = 128
    gray = np.where(gray > threshold, 255, 0).astype(np.uint8)

    # Удаление пустых границ
    while np.sum(gray[0]) == 0:
        gray = gray[1:]
    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, axis=1)
    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]
    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, axis=1)

    rows, cols = gray.shape

    # Масштабирование до 20x20
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
    gray = np.array(Image.fromarray(gray).resize((cols, rows), Image.Resampling.LANCZOS))

    # Добавление паддингов до 28x28
    colsPadding = (int(math.ceil((28 - cols)/2)), int(math.floor((28 - cols)/2)))
    rowsPadding = (int(math.ceil((28 - rows)/2)), int(math.floor((28 - rows)/2)))
    gray = np.pad(gray, (rowsPadding, colsPadding), 'constant', constant_values=0)

    # Ограничение сдвига
    shiftx, shifty = getBestShift(gray)
    max_shift_x = cols // 2
    max_shift_y = rows // 2
    shiftx = np.clip(shiftx, -max_shift_x, max_shift_x)
    shifty = np.clip(shifty, -max_shift_y, max_shift_y)

    # Применение сдвига
    gray = shift(gray, shiftx, shifty, rows, cols)

    # Сохранение и вывод
    processed_img = Image.fromarray(gray.astype(np.uint8))
    processed_img.save(f"gray_{uploaded_file.name}")
    img = gray / 255.0
    img = img.reshape(-1, 28, 28, 1)
    return str(np.argmax(model.predict(img)))

# Заголовок приложения
st.markdown('">Распознавание AZOV 222 рукописных цифр</h1>', unsafe_allow_html=True)

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
        st.markdown(f'">Предсказанная цифра: {result}</p>', unsafe_allow_html=True)
        #st.markdown(f'<p style="color: white;">Уверенность в предсказании: {conf}</p>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Произошла ошибка: {e}")



