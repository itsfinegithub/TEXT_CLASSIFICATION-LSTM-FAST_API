from fastapi import FastAPI
import pickle
from preprocess import text_cleaning
import logging
import time
import numpy as np
import tensorflow 


app = FastAPI()
# loading tfidf model
tokenization  = pickle.load(open('tokenizer.pickle', 'rb'))
model = tensorflow.keras.models.load_model('lstm_model.h5')

# it is the basic configuration it will create stream handler
logging.basicConfig(filename='logfile.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


@app.get('/')
def home():
    return {'welcome to cyber security classification model'}


@app.post('/cyber_security_prediction/')
def predict(text):

    start = time.perf_counter()
    #  proprocess text

    clean_text = text_cleaning(text)
    logging.info('preprocess done')

    # converting text into sequence of numbers
    tokenized_text = tokenization.texts_to_sequences(np.array([clean_text]))

    # pad_sequences is used to ensure that all sequences in a list have the same length
    vector = tensorflow.keras.preprocessing.sequence.pad_sequences(tokenized_text, maxlen=50)
    logging.info('converted text into numbers')

    # prediction
    prediction = model.predict(vector)
    logging.info(f'prediction value is {prediction}')

    # condition for prediction
    if prediction <= 0.5:
        msg = 'not cyber security tweets'
    else:
        msg = 'cyber security tweets'

    logging.info(f'text is {msg}')
    end = time.perf_counter()

    #task finishing time
    logging.info(f'finished time is {end}-{start}')

    # return prediction
    return {'predicted message is': msg,
    'probability' : float(prediction*100)}