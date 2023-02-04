from tensorflow import keras
import tensorflow as tf
import numpy as np

def predict(model, image_file):
   image_size = 28
   img = keras.preprocessing.image.load_img(image_file,target_size=(image_size, image_size), color_mode='grayscale')
   img_arr = np.expand_dims(img, axis=0)
   img_arr = 1 - img_arr/255.0
   img_arr = img_arr.reshape((1, 28, 28, 1))

   result = model.predict_classes([img_arr])
   return result[0]

model = tf.keras.models.load_model('mnist.h5')
print(predict(model, '2.png'))