import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow import keras
import tensorflow as tf
import numpy as np

def cnn_digits_predict(model, image_file):
   image_size = 28
   img = keras.preprocessing.image.load_img(image_file,
target_size=(image_size, image_size), color_mode='grayscale')
   img_arr = np.expand_dims(img, axis=0)
   img_arr = 1 - img_arr/255.0
   img_arr = img_arr.reshape((1, 28, 28, 1))

   result = model.predict_classes([img_arr])
   return result[0]

model = tf.keras.models.load_model('cnn_digits_28x28.h5')
print(cnn_digits_predict(model, '1.png'))