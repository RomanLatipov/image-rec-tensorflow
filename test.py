from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import os, cv2, tensorflow as tf, numpy as np

img = cv2.imread('happytest.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

resize = tf.image.resize(img, (128, 128))
# plt.imshow(resize.numpy().astype(int))

new_model = load_model(os.path.join('models', 'happysadmodel.h5'))
yhat = new_model.predict(np.expand_dims(resize/255, 0))

if yhat > 0.5:
    print('Predicted class is sad')
else:
    print('Predicted class is sad')