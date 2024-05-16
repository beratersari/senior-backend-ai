from architecture import BertForSTS
import torch
from collections import defaultdict
from transformers import BertTokenizer
import json
import sys
sys.path.append('./../')
from ranking_measures import measures
device =  "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_similarity(sentence_pair, model):
  test_input = tokenizer(sentence_pair, padding='max_length', max_length = 512, truncation=True, return_tensors="pt").to(device)
  test_input['input_ids'] = test_input['input_ids']
  test_input['attention_mask'] = test_input['attention_mask']
  del test_input['token_type_ids']
  output = model(test_input)
  print(output.shape)
  sim = torch.nn.functional.cosine_similarity(output[0], output[1], dim=0).item()
  return sim


PATH = 'model.pth'
model = BertForSTS()
model.load_state_dict(torch.load(PATH))
model.eval()
model.to(device)
sentence_pair = ["I am a student", "I am a teacher"]
similarity = predict_similarity(sentence_pair, model)
print(f"Similarity between '{sentence_pair[0]}' and '{sentence_pair[1]}' is {similarity:.4f}")


def calculate_similarity_matrix(abstracts):
  similarity_matrix = defaultdict(list)
  for i, (text_id1, emb1) in enumerate(abstracts):
    for j, (text_id2, emb2) in enumerate(abstracts):
      similarity_matrix[text_id1].append((predict_similarity([emb1,emb2],model), text_id2))
  return similarity_matrix

with open('data_new.json', encoding='utf-8') as f:
  data = json.load(f)
abstarcts =[]
text_map = {}
for item in data:
    abstarcts.append( (item["id"], item["abstract"] + " ".join(item["keywords"]) ))
    text_map[item["id"]] =item["abstract"] + " ".join(item["keywords"])
print(len(abstarcts))
similarity_matrix = calculate_similarity_matrix(abstarcts)


# Load the data from JSONL file
data = []
with open("posneg.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))
for key in similarity_matrix.keys():
    similarity_matrix[key].sort()
    similarity_matrix[key] = similarity_matrix[key][::-1]
    print(similarity_matrix[key][:10])
    print("")
def calculate_map(data, text_list,sim_matrix):
    total_reciprocal_rank = 0
    num_queries = len(data)


    for query_data in data:
        query_id = query_data["query"]
        pos_examples = query_data["pos"]
        sim_matrix[query_id].sort()
        sim_matrix[query_id]=sim_matrix[query_id][::-1]
        hypotesis = [sim_matrix[query_id][i][1] for i in range(1,len(sim_matrix[query_id]))]

        total_reciprocal_rank += measures.find_precision_k(pos_examples, hypotesis,5)





    # Compute Mean Reciprocal Rank
    mrr_score = total_reciprocal_rank / num_queries
    return mrr_score

# Calculate MAP
map_score = calculate_map(data, text_map,similarity_matrix)
print("Mean Average Precision (MAP):", map_score)