import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json
import numpy as np
from os import system

input_size=40

fh=open("corpus.txt")
data=fh.read().lower()
tokenizer=Tokenizer(num_words=37, oov_token="UKC", char_level=True)
tokenizer.fit_on_texts(data)
c_index=tokenizer.word_index

def clear() :
    _=system("cls")

clear()

test_data=input("Input char sequence of less than 40 chars:").lower()
seq=np.array(tokenizer.texts_to_sequences(test_data)).reshape(1,len(test_data))
padded=pad_sequences(seq, padding="pre", truncating="post", maxlen=input_size)
length=int(input("Enter desired length:"))

json_file = open('model_w.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_w.h5")


pred_c=""
temp_s=test_data
for _ in range(length) :
    padded=np.array(tokenizer.texts_to_sequences(temp_s)).reshape(1,len(temp_s))
    padded=pad_sequences(padded, padding="pre", truncating="pre", maxlen=20)
    pred=loaded_model.predict_classes(padded)
    for ch,index in c_index.items() :
        if index==pred :
            pred_c=ch
            break
    temp_s=temp_s+pred_c

clear()

print(temp_s)
