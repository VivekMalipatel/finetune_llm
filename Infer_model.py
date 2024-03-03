from finetune import TriageDataset,Config
from transformers import AutoTokenizer,AutoModel
from torch.utils.data import DataLoader
import torch
import os
import pandas as pd

Config.BERT_PATH = 'bert_finetuned'

tokenizer = AutoTokenizer.from_pretrained(Config.BERT_PATH)

data = {"Body": ["hello vivekanand thank you for your interest in software engineer coop month and the time you invested in applying for the opening we regret to inform you that you were not selected for further consideration your resume will remain active in our talent management system in accordance with applicable law and we encourage you to continue to explore additional opportunity at amd please be sure to keep your candidate profile updated at our career opportunity page thank you again for your interest in amd sincerely the amd talent acquisition team this message wa sent to if you dont want to receive these email from this company in the future please go to attachment n advanced micro device santa clara california n"], "ENCODE_CAT": [1]}
df = pd.DataFrame(data)

training_set = TriageDataset(df, tokenizer, Config.MAX_LEN)
print(training_set)

train_params = {'batch_size': Config.TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

training_loader = DataLoader(training_set, **train_params)

model = AutoModel.from_pretrained(Config.BERT_PATH)

for _, data in enumerate(training_loader, 0):
    ids = data['ids'].to(Config.device, dtype=torch.long)
    mask = data['mask'].to(Config.device, dtype=torch.long)
    outputs = model(ids, mask)
    big_val, big_idx = torch.max(outputs.data, dim=1)
    print(outputs)
    print(big_idx)