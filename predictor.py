import cv2
import numpy as np
import tensorflow as tf

from PIL import Image


def train() -> None:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) =  mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10)

    model.save('number.model')


def load() -> None:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) =  mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    return tf.keras.models.load_model('number.model')


def show_img(img) -> None:
    cv2.imshow('image', img)
    cv2.waitKey(0)


def detect(img):
    SCALE = 3
    THICK = 5
    WHITE = (255, 255, 255)

    digits = []
    
    for digit in map(str, range(10)):
        (width, height), bline = cv2.getTextSize(digit, cv2.FONT_HERSHEY_SIMPLEX,
                                                SCALE, THICK)
        digits.append(np.zeros((height + bline, width), np.uint8))
        cv2.putText(digits[-1], digit, (0, height), cv2.FONT_HERSHEY_SIMPLEX,
                    SCALE, WHITE, THICK)
        x0, y0, w, h = cv2.boundingRect(digits[-1])
        digits[-1] = digits[-1][y0:y0+h, x0:x0+w]
    
    percent_white_pix = 0
    digit = -1
    
    for i, d in enumerate(digits):
        scaled_img = cv2.resize(img, d.shape[:2][::-1])
        bitwise = cv2.bitwise_and(d, cv2.bitwise_xor(scaled_img, d))
        before = np.sum(d == 255)
        matching = 100 - (np.sum(bitwise == 255) / before * 100)
        
        if percent_white_pix < matching:
            percent_white_pix = matching
            digit = i
    
    return digit


def find_number_on_img(img):
    img = cv2.imread(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    msk = cv2.inRange(hsv, np.array([0, 155, 165]), np.array([205, 255, 255]))
    
    count: int = 0
    for i in range(len(msk)):
        for j in range(len(msk[i])):
            if msk[i][j] != 0:
                count += 1
    
    if count == 0:
        msk = cv2.inRange(hsv, np.array([0, 0, 165]), np.array([205, 255, 255]))
    
    count = 0
    for i in range(len(msk)):
        for j in range(len(msk[i])):
            if msk[i][j] != 0:
                count += 1
                
    if count > 150000:
        msk = cv2.inRange(hsv, np.array([0, 0, 45]), np.array([205, 255, 255]))
        krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        dlt = cv2.dilate(msk, krn, iterations=1)
        thr = 255 - cv2.bitwise_and(dlt, msk)
        rz = cv2.resize(thr, (28, 28), fx = 1, fy = 1)
        
        return rz
        
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dlt = cv2.dilate(msk, krn, iterations=1)
    thr = cv2.bitwise_and(dlt, msk)
    rz = cv2.resize(thr, (28, 28), fx = 1, fy = 1)
    
    return rz


def find_number(img):    
    img = cv2.imread(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 75 200
    # 40 30 
    msk = cv2.Canny(hsv, 40, 30)
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dlt = cv2.dilate(msk, krn, iterations=1)
    thr = cv2.bitwise_and(dlt, msk)

    
    contours, hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 30:
            brect = cv2.boundingRect(cnt)
            x,y,w,h = brect

    try:
        crop = thr[y-10:y+h+10, x-10:x+w+10]
    except:
        crop = thr
        
    rz = cv2.resize(crop, (28, 28))
    
    cv2.destroyAllWindows()
    
    return rz


def predict_img(img):
    model = load()
    test = cv2.imread(img)
    hsv = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    msk = cv2.Canny(hsv, 40, 30)
    
    count: int = 0
    
    for i in range(len(msk)):
        for j in range(len(msk[i])):
            if msk[i][j] != 0:
                count += 1
    
    if count < 1300:
        try:
            rz_image = find_number(img)
        except:
            print("Error!")
    else:
        rz_image = find_number_on_img(img)

    image = np.array([rz_image])
    prediction = model.predict(image)
    
    rz_image = Image.fromarray(rz_image)
    
    return np.argmax(prediction), rz_image


def main() -> None:
    print(predict_img(r"numbers\number18.png"))


if __name__ == "__main__":
    main()