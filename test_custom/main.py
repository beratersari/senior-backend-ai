import json

# load the json file with explicit encoding
with open('data_new.json', encoding='utf-8') as f:
    data = json.load(f)

counter = 0
for item in data:
    print(item["id"])
    print(item["project_name"])
    print(item["abstract"])
    print(*item["keywords"], sep=", ")
    print("*****"*25)
