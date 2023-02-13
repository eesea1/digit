from tensorflow import keras
import tensorflow as tf
import numpy as np


def predict(model, img):
   img = img.resize((28, 28))
   img = img.convert('L')
   img = np.array(img)
   img = img.reshape(1, 28, 28, 1)
   img = img / 255.0
   res = model.predict([img])[0]
   return np.argmax(res), max(res)

model = tf.keras.models.load_model('mnist.h5')
print(predict(model, '2.png'))