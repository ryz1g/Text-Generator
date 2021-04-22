import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
import numpy as np
import time
import matplotlib.pyplot as plt
t0=time.time()

fh=open("corpus.txt")
data=fh.read().lower()
tokenizer=Tokenizer(num_words=37, oov_token="UKC", char_level=True)
tokenizer.fit_on_texts(data)
c_index=tokenizer.word_index
seq_data=tokenizer.texts_to_sequences(data)

class myCall(tf.keras.callbacks.Callback) :
    def on_epoch_end(self, epoch, logs={}) :
        if(logs.get('accuracy')>0.87) :
            self.model.stop_training=True

def plot_graphs(history, string):
    #plt.plot(history.history["loss"])
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

callbacks=myCall()

#hyperparameters
input_size=40
emb_dim=10
op=Adam(lr=0.000001)
num_epochs=8
b_size=128

temp,hmm=([],[])
flag=0
for c in range(len(seq_data)):
    if c+input_size+1>len(seq_data) :
        break
    temp.append(seq_data[c:c+input_size+1])

for t in temp:
    for i in range(2,input_size+2):
        hmm.append(t[:i])

padded=pad_sequences(hmm, padding="pre", truncating="post", maxlen=input_size+1)

xs,ys=([],[])

for i in padded:
    xs.append(i[:-1])
    ys.append(i[-1:])

ys=tf.keras.utils.to_categorical(ys, num_classes=37)
xs=np.array(xs).reshape(len(xs),input_size)
ys=np.array(ys).reshape(len(ys),37)

print("Time take to Load Model:", end=" ")
print(time.time()-t0)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(37, emb_dim, input_length=input_size))
model.add(tf.keras.layers.Conv1D(2048, 3, activation="relu"))
model.add(tf.keras.layers.GlobalMaxPooling1D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(37, activation="softmax"))
model.load_weights("model_w.h5")
model.compile(loss="categorical_crossentropy", optimizer=op, metrics=["accuracy"])
history=model.fit(xs, ys, epochs=num_epochs, batch_size=b_size, verbose=1, callbacks=[callbacks], shuffle=True)

model.summary()
plot_graphs(history, 'loss')

ti=input("save weights?(Y/N)")
if ti=="Y" :
    model_json=model.to_json()
    with open("model_w.json", "w") as json_file :
        json_file.write(model_json)
    model.save_weights("model_w.h5")
