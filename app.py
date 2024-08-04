import tensorflow as tf, numpy as np
from matplotlib import pyplot as plt

# print(tf.cofig.experimental.list_pysical_devices('GPU'))

# img = cv2.imread(os.path.join('happy', '154006829.jpg'))
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

data = tf.keras.utils.image_dataset_from_directory('data')
print(data)