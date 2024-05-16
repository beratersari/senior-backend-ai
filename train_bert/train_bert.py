import torch
from transformers.utils.notebook import format_time

from data_utils import prepare_data
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from architecture import BertForSTS, CosineSimilarityLoss
from dataset import STSBDataset, collate_fn
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, models
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import datetime
import random
import numpy as np
import pandas as pd
def train(train_dataloader, validation_dataloader, model, device):
  seed_val = 42
  optimizer = AdamW(model.parameters(),
                    lr=1e-6)
  epochs = 8
  # Total number of training steps is [number of batches] x [number of epochs].
  total_steps = len(train_dataloader) * epochs
  scheduler = get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=2,
                                              num_training_steps=total_steps)
  criterion = CosineSimilarityLoss()
  criterion = criterion.cuda()
  random.seed(seed_val)
  torch.manual_seed(seed_val)
  # We'll store a number of quantities such as training and validation loss,
  # validation accuracy, and timings.
  training_stats = []
  total_t0 = time.time()
  for epoch_i in range(0, epochs):
      t0 = time.time()
      total_train_loss = 0
      model.train()
      # For each batch of training data...
      for train_data, train_label in tqdm(train_dataloader):
          train_data['input_ids'] = train_data['input_ids'].to(device)
          train_data['attention_mask'] = train_data['attention_mask'].to(device)
          train_data = collate_fn(train_data)
          model.zero_grad()
          output = [model(feature) for feature in train_data]
          loss = criterion(output, train_label.to(device))
          total_train_loss += loss.item()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          optimizer.step()
          scheduler.step()

      # Calculate the average loss over all of the batches.
      avg_train_loss = total_train_loss / len(train_dataloader)
      # Measure how long this epoch took.
      training_time = format_time(time.time() - t0)
      t0 = time.time()
      model.eval()
      total_eval_accuracy = 0
      total_eval_loss = 0
      nb_eval_steps = 0
      # Evaluate data for one epoch
      for val_data, val_label in tqdm(validation_dataloader):
          val_data['input_ids'] = val_data['input_ids'].to(device)
          val_data['attention_mask'] = val_data['attention_mask'].to(device)
          val_data = collate_fn(val_data)
          with torch.no_grad():
              output = [model(feature) for feature in val_data]
          loss = criterion(output, val_label.to(device))
          total_eval_loss += loss.item()
      # Calculate the average loss over all of the batches.
      avg_val_loss = total_eval_loss / len(validation_dataloader)
      # Measure how long the validation run took.
      validation_time = format_time(time.time() - t0)
      print ("Epoch {}:".format(epoch_i + 1))
      print(f"  Training Loss: {avg_train_loss}")
      print(f"  Validation Loss: {avg_val_loss}")
      print(f"  Training epcoh took: {training_time}")
      print(f"  Validation epoch took: {validation_time}")
      # Record all statistics from this epoch.
      training_stats.append(
          {
              'epoch': epoch_i + 1,
              'Training Loss': avg_train_loss,
              'Valid. Loss': avg_val_loss,
              'Training Time': training_time,
              'Validation Time': validation_time
          }
      )
  return model, training_stats


def main():
    pairs, labels = prepare_data("data_new.json", "posneg.jsonl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #splite train val in a dictionary
    train_val_dict = {}
    train_val_dict['train'] = {}
    train_val_dict['val'] = {}
    train_val_dict['train']['pairs'], train_val_dict['val']['pairs'], train_val_dict['train']['labels'], train_val_dict['val']['labels'] = train_test_split(pairs, labels, test_size=0.2)
    model = BertForSTS()
    model.to(device)
    train_ds = STSBDataset(train_val_dict['train']['pairs'], train_val_dict['train']['labels'])
    val_ds = STSBDataset(train_val_dict['val']['pairs'], train_val_dict['val']['labels'])
    batch_size = 8

    train_dataloader = DataLoader(
        train_ds,  # The training samples.
        num_workers=4,
        batch_size=batch_size,  # Use this batch size.
        shuffle=True  # Select samples randomly for each batch
    )

    validation_dataloader = DataLoader(
        val_ds,
        num_workers=4,
        batch_size=batch_size  # Use the same batch size
    )
    print("Training started.")
    trained_model, training_stats = train(train_dataloader, validation_dataloader, model, device)
    print("")
    print("Training complete!")
    #save model
    torch.save(trained_model.state_dict(), 'model.pth')
    print("Model saved")



if __name__ == '__main__':
    main()