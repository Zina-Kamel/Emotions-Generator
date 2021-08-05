import numpy as np
from numpy.lib.function_base import interp
import tensorflow as tf
import keras
from  keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from data import *

def load_model():
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=".\model\emotion-tdetector.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details

model, input_details, output_details = load_model()
def run_inference(interpreter, sentence, input_details, output_details):
    input_data = np.array(sentence, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

df_train = pd.read_csv('data/test.txt', 
                       header =None, 
                       sep =';', 
                       names = ['Text','Sentiment'], 
                       encoding='utf-8')

def get_emotion(emotion):
    labels = {
        "sadness":0,
        "love":1,
        "anger":2,
        "surprise":3,
        "joy":4,
        "fear":5
    }
    for key,val in labels.items():
        if (val==emotion):
            return key

tokenizer=Tokenizer(15212,lower=True,oov_token='UNK')
tokenizer.fit_on_texts(df_train['Text'])

def predict(sentence):
  sentences=[]
  sentences.append(sentence)
  sentence_seq=tokenizer.texts_to_sequences(sentences)
  sentence_padded=pad_sequences(sentence_seq,maxlen=80,padding='post')
  output = run_inference(model, sentence_padded, input_details, output_details )
  res=get_emotion(output.argmax())
  return output, res

