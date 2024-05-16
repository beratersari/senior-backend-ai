import json
def prepare_data(json_path, jsonl_path):
    # load the json file with explicit encoding
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    abstracts = []
    text_map = {}
    for item in data:
        abstracts.append(item["abstract"])
        text_map[item["id"]] = item["abstract"] + " ".join(item["keywords"])
    # load the json file with explicit encoding
    similarity_data = []
    with open("posneg.jsonl", "r") as f:
        for line in f:
            similarity_data.append(json.loads(line))
    similarity_matrix = [ [0 for i in range(len(abstracts))] for j in range(len(abstracts) ) ]
    for item in similarity_data:
        query_id = item["query"]
        pos_examples = item["pos"]
        for pos_example in pos_examples:
            similarity_matrix[query_id][pos_example] = 1
            similarity_matrix[pos_example][query_id] = 1
    pairs = []
    labels = []
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix)):
            if i != j:  # Exclude self-comparisons
                pairs.append((text_map[i], text_map[j]))
                labels.append(similarity_matrix[i][j])
    return pairs, labels