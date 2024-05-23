import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
import time
import json
import numpy as np
from collections import defaultdict
import sys
sys.path.append('./../')
from ranking_measures import measures
# Load pre-trained BERT tokenizer and model
from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, \
    DistilBertTokenizer, DistilBertModel, OpenAIGPTTokenizer, OpenAIGPTModel, \
    GPT2Tokenizer, GPT2Model, XLNetTokenizer, XLNetModel, T5Tokenizer, T5Model
model_type = 'xlnet'
# DistilBERT
if model_type == 'distilbert':
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
elif model_type == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
elif model_type == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
elif model_type == 'xlnet':
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    model = XLNetModel.from_pretrained('xlnet-base-cased')




#tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#model = RobertaModel.from_pretrained('roberta-base')
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs['last_hidden_state'][:,0,:].numpy().reshape(-1)

def cosine_similarity_bert(emb1, emb2):
    similarity = 1 - cosine(emb1, emb2)
    return similarity
def extract_embeddings(texts):
	embeds=[]
	for text_id in texts.keys():
		start = time.time()
		embeds.append((text_id,get_bert_embedding(texts[text_id])))
		end =time.time() 
		print(end- start,"seconds")
	return embeds
def calculate_similarity_matrix(embeds):
    similarity_matrix = defaultdict(list)
    for i, (text_id1, emb1) in enumerate(embeds):
        for j, (text_id2,emb2) in enumerate(embeds):
            similarity_matrix[text_id1].append((cosine_similarity_bert(emb1, emb2),text_id2))
    return similarity_matrix

def print_top_similar_texts(similarity_matrix, text_list, k=6):
    for i in text_list.keys():
        print(f"Top {k} similar texts for '{text_list[i]}':")
        
        similarities = similarity_matrix[i]
        # Sort indices based on similarity scores (excluding self-similarity)
        similarities.sort()
        similarities= similarities[::-1]
        similarities=similarities[:k]
        print(i)
        print("similarity")
        for idx in similarities:
        	print(idx[1])
        print("*"*25)

# load the json file with explicit encoding
with open('data_new.json', encoding='utf-8') as f:
    data = json.load(f)

counter = 0
abstarcts =[]
text_map = {}
for item in data:
    #print(item["id"])
    #print(item["project_name"])
    #print(item["abstract"])
    #print(*item["keywords"], sep=", ")
    #print("*****"*25)
    abstarcts.append( item["abstract"] + " ".join(item["keywords"]))
    text_map[item["id"]] =item["abstract"] + " ".join(item["keywords"])
embeds = extract_embeddings(text_map)
for i in embeds:
    print(i[1].shape)
    i_str = [str(x) for x in i[1]]
    print(len( " ".join(i_str)))
    print("*"*25)

sim_matrix = calculate_similarity_matrix(embeds)
print_top_similar_texts(sim_matrix, text_map)








# Load the data from JSONL file
data = []
with open("posneg.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

def calculate_map(data, text_list,sim_matrix):
    total_reciprocal_rank = 0
    num_queries = len(data)


    for query_data in data:
        query_id = query_data["query"]
        pos_examples = query_data["pos"]
        neg_examples = query_data["neg"]
        query_text = text_list[query_id]
        sim_matrix[query_id].sort()
        sim_matrix[query_id]=sim_matrix[query_id][::-1]
        hypotesis = [sim_matrix[query_id][i][1] for i in range(1,len(sim_matrix[query_id]))]



        total_reciprocal_rank += measures.find_precision_k(pos_examples,hypotesis,5)

    # Compute Mean Reciprocal Rank
    mrr_score = total_reciprocal_rank / num_queries
    return mrr_score

# Calculate MAP
map_score = calculate_map(data, text_map,sim_matrix)
print("Mean Average Precision (MAP):", map_score)

