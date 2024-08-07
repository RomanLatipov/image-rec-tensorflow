from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from app import *
from model import model

pre = Precision()
rec = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    x, y = batch
    yhat = model.predict(x)
    pre.update_state(y, yhat)
    rec.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision:{pre.result()}, Recall:{rec.result()}, Accuracy:{acc.result()}')
