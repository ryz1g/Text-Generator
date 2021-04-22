import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json
import numpy as np

fh=open("corpus.txt")
data=fh.read().lower().split("\n")

tokenizer=Tokenizer(oov_token="UKC")
tokenizer.fit_on_texts(data)
w_index=tokenizer.word_index

it=input("Input word sequence of less than 11 words:")
test_data=it.lower().split(" ")
seq=np.array(tokenizer.texts_to_sequences(test_data)).reshape(1,len(test_data))
padded=pad_sequences(seq, padding="pre", truncating="post", maxlen=11)

length=int(input("Enter desired length:"))

json_file = open('model_w.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_w.h5")

temp_s=it
pred_w=""
for _ in range(length) :
    temp_l=temp_s.split(" ")
    padded=np.array(tokenizer.texts_to_sequences(temp_l)).reshape(1,len(temp_l))
    padded=pad_sequences(padded, padding="pre", truncating="post", maxlen=14)
    pred=loaded_model.predict_classes(padded)
    for w,index in w_index.items() :
        if index==pred :
            pred_w=w
            break
    temp_s=temp_s+" "+pred_w

print(temp_s)
