import cv2
import pytesseract
import keras
from keras.models import load_model
from keras.datasets import mnist
from keras.models import Sequential
import numpy as np

model = load_model('mnist.h5')

def predict_digit(img):
    # изменение рзмера изобржений на 28x28
    img = img.resize((28,28))
    # конвертируем rgb в grayscale
    img = img.convert('L')
    img = np.array(img)
    # изменение размерности для поддержки модели ввода и нормализации
    img = img.reshape(1,28,28,1)
    img = img/255.0
    # предстказание цифры
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

img = cv2.imread('1.png', 0)
thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

data = pytesseract.image_to_string(thresh, lang='eng',config='--psm 6')
print(data)
