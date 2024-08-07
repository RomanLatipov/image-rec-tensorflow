import tensorflow as tf, numpy as np, cv2
from matplotlib import pyplot as plt

# print(tf.cofig.experimental.list_pysical_devices('GPU'))

# img = cv2.imread(os.path.join('happy', '154006829.jpg'))
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

data = tf.keras.utils.image_dataset_from_directory('data', batch_size=4, image_size=(128,128))
# print(data)
data_iterator = data.as_numpy_iterator()
# print(data_iterator)
batch = data_iterator.next()
# print(batch)

# fig, ax = plt.subplots(ncols=4, figsize=(10,10))
# for idx, img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img.astype(int))
#     ax[idx].title.set_text(batch[1][idx])
# plt.show()

data = data.map(lambda x, y: (x/255, y))
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# for idx, img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img)
#     ax[idx].title.set_text(batch[1][idx])
# plt.show()

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)+1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)
