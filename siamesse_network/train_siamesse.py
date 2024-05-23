from operator import itemgetter
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Lambda
from keras.optimizers import Adam
from keras import backend as K
from model import SiameseBiLSTM
import json
from config import siamese_config
from inputHandler import word_embed_meta_data, create_test_data
import warnings
warnings.filterwarnings('ignore')
#igore tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data from JSON files
text_map = {}
similarity_matrix = None

# Load text map
with open('data_new.json', encoding='utf-8') as f:
    data = json.load(f)
    for item in data:
        text_map[item["id"]] = item["abstract"] + " ".join(item["keywords"])

# Load similarity matrix
with open("posneg.jsonl", "r") as f:
    data = []
    for line in f:
        data.append(json.loads(line))
    n = len(text_map)
    similarity_matrix = np.zeros((n, n))
    for query_data in data:
        query_id = query_data["query"]
        pos_examples = query_data["pos"]
        for pos_example in pos_examples:
            similarity_matrix[query_id][pos_example] = 1

sentences1= []
sentences2 = []
labels = []
for i in range(len(similarity_matrix)):
    for j in range(len(similarity_matrix)):
        if i != j:  # Exclude self-comparisons
            sentences1.append(text_map[i])
            sentences2.append(text_map[j])
            labels.append(similarity_matrix[i][j])

def get_tokenizer():
    global sentences1, sentences2
    tokenizer, embedding_matrix = word_embed_meta_data(sentences1 + sentences2, siamese_config['EMBEDDING_DIM'])
    return tokenizer, embedding_matrix


def main():
    global sentences1, sentences2, labels
    tokenizer, embedding_matrix = get_tokenizer()
    embedding_meta_data = {
        'tokenizer': tokenizer,
        'embedding_matrix': embedding_matrix
    }

    ## creating sentence pairs
    sentences_pair = [(x1, x2) for x1, x2 in zip(sentences1, sentences2)]
    del sentences1
    del sentences2
    class Configuration(object):
        """Dump stuff here"""

    CONFIG = Configuration()

    CONFIG.embedding_dim = siamese_config['EMBEDDING_DIM']
    CONFIG.max_sequence_length = siamese_config['MAX_SEQUENCE_LENGTH']
    CONFIG.number_lstm_units = siamese_config['NUMBER_LSTM']
    CONFIG.rate_drop_lstm = siamese_config['RATE_DROP_LSTM']
    CONFIG.number_dense_units = siamese_config['NUMBER_DENSE_UNITS']
    CONFIG.activation_function = siamese_config['ACTIVATION_FUNCTION']
    CONFIG.rate_drop_dense = siamese_config['RATE_DROP_DENSE']
    CONFIG.validation_split_ratio = siamese_config['VALIDATION_SPLIT']

    siamese = SiameseBiLSTM(CONFIG.embedding_dim , CONFIG.max_sequence_length, CONFIG.number_lstm_units , CONFIG.number_dense_units, CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense, CONFIG.activation_function, CONFIG.validation_split_ratio)

    model = siamese.train_model(sentences_pair, labels, embedding_meta_data, model_save_directory='./')
    model.save('siamese_model.h5')
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    test_sentence_pairs = [('Scalability is an attribute and a characteristic of systems which describes the ability of the system to grow and manage increased demand of the systems. The scalability problems in web applications shows up more specifically in recent years since the number of users and as a result, number of requests have increased considerably. The big companies deal with this problem using various technologies and techniques. In this project, we used microservice architecture, containerization technologies and relational, non-relational DBMS’ to create a scalable but also highly available course registration system.",',
                            "In this project, we introduce FOCUS as web application that hosts many necessary features that clinicians will use. Clinicians can register to the FOCUS and create projects. They can invite other clinicians and collect data about own project."), (
    'How many times a day do a clocks hands overlap?',
    'What does it mean that every time I look at the clock the numbers are the same?')]
    test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer, test_sentence_pairs,
                                                              siamese_config['MAX_SEQUENCE_LENGTH'])

    preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
    results = [(x, y, z) for (x, y), z in zip(test_sentence_pairs, preds)]
    results.sort(key=itemgetter(2), reverse=True)
    print(results)

if __name__ == '__main__':
    main()