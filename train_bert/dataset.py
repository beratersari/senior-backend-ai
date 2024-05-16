import torch
from transformers import BertTokenizer
class STSBDataset(torch.utils.data.Dataset):

    def __init__(self, pairs, scores):
        # Normalize the similarity scores in the dataset
        similarity_scores = [i for i in scores]
        self.normalized_similarity_scores = [i/1.0 for i in similarity_scores]
        self.first_sentences = [i[0] for i in pairs]
        self.second_sentences = [i[1] for i in pairs]
        self.concatenated_sentences = [[str(x), str(y)] for x,y in   zip(self.first_sentences, self.second_sentences)]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.concatenated_sentences)

    def get_batch_labels(self, idx):
        return torch.tensor(self.normalized_similarity_scores[idx])

    def get_batch_texts(self, idx):
        return self.tokenizer(self.concatenated_sentences[idx], padding='max_length', max_length=512, truncation=True, return_tensors="pt")

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


def collate_fn(texts):
    input_ids = texts['input_ids']
    attention_masks = texts['attention_mask']
    features = [{'input_ids': input_id, 'attention_mask': attention_mask}
                for input_id, attention_mask in zip(input_ids, attention_masks)]
    return features