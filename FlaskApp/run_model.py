import random

from architecture import BertForSTS
import torch
from collections import defaultdict
from transformers import BertTokenizer
import json
import sys
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
PATH = 'model.pth'
model = BertForSTS()
model.load_state_dict(torch.load(PATH))
model.eval()

def get_top_similars(target_embedding, other_embeddings):
    similarity_scores = []
    #convert tensor target_embedding and other_embeddings
    target_embedding = torch.tensor(target_embedding)
    other_embeddings = [(id, torch.tensor(embeddings)) for id, embeddings in other_embeddings]
    for id, embeddings in other_embeddings:
        similarity = torch.nn.functional.cosine_similarity(target_embedding, embeddings, dim=0).item()
        similarity_scores.append((similarity, id))
    similarity_scores.sort(reverse=True)
    #select random 5 from top 10
    top_similars = similarity_scores[:10]
    random_top_similars = random.sample(top_similars, 5)
    return random_top_similars


def get_embeddings(sentence):
  test_input = tokenizer([sentence,sentence], padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
  test_input['input_ids'] = test_input['input_ids']
  test_input['attention_mask'] = test_input['attention_mask']
  del test_input['token_type_ids']
  output = model(test_input)
  output0 = output[0].detach().numpy().tolist()
  output0 = [str(i) for i in output0]

  return " ".join(output0)

