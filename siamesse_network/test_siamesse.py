from operator import itemgetter
from keras.models import load_model
import sys
sys.path.append('./../')
import tensorflow as tf
from ranking_measures import measures
from config import siamese_config
from inputHandler import create_test_data
from train_siamesse import get_tokenizer
import json
from collections import defaultdict
best_model_path = 'siamese_model.h5'
model = load_model(best_model_path)
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
# load the json file with explicit encoding
with open('data_new.json', encoding='utf-8') as f:
    data = json.load(f)

counter = 0
abstarcts =[]
text_map = {}
text_list = {}
for item in data:
    #print(item["id"])
    #print(item["project_name"])
    #print(item["abstract"])
    #print(*item["keywords"], sep=", ")
    #print("*****"*25)
    abstarcts.append(item["abstract"] + " ".join(item["keywords"]))
    text_map[item["abstract"] + " ".join(item["keywords"])] = item["id"]
    text_list[item["id"]] = item["abstract"] + " ".join(item["keywords"])

test_sentence_pairs = []
for i in range(len(abstarcts)):
    for j in range(len(abstarcts)):
        if i != j:  # Exclude self-comparisons
            test_sentence_pairs.append((abstarcts[i], abstarcts[j]))

tokenizer, embedding_matrix = get_tokenizer()
test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,test_sentence_pairs,  siamese_config['MAX_SEQUENCE_LENGTH'])

preds = list(model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
results = [(x, y, z) for (x, y), z in zip(test_sentence_pairs, preds)]
results.sort( reverse=True)
sim_matrix = defaultdict(list)
for x,y,z in results:
    sim_matrix[text_map[x]].append((z,text_map[y]))

# Load the data from JSONL file
data = []
with open("posneg.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))


def calculate_map(data, text_list, sim_matrix):
    total_reciprocal_rank = 0
    num_queries = len(data)

    for query_data in data:
        query_id = query_data["query"]
        pos_examples = query_data["pos"]
        neg_examples = query_data["neg"]
        query_text = text_list[query_id]
        sim_matrix[query_id].sort()
        sim_matrix[query_id] = sim_matrix[query_id][::-1]
        hypotesis = [sim_matrix[query_id][i][1] for i in range(1, len(sim_matrix[query_id]))]

        total_reciprocal_rank += measures.find_precision_k(pos_examples, hypotesis,5)

    # Compute Mean Reciprocal Rank
    mrr_score = total_reciprocal_rank / num_queries
    return mrr_score


# Calculate MAP
map_score = calculate_map(data, text_list, sim_matrix)
print("Mean Average Precision (MAP):", map_score)