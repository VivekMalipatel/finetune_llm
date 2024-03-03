from finetune import TriageDataset,Config, BERTClassifier
from transformers import BertTokenizer, logging
from torch.utils.data import DataLoader
import torch
import pandas as pd

logging.set_verbosity_error()

Config.BERT_PATH = 'bert_finetuned'

tokenizer = BertTokenizer.from_pretrained(Config.BERT_PATH)

data = {"Body": ["We appreciate your interest in joining the Seismic team. We know how time consuming it can be searching for a new opportunity, so we really mean it when we say thanks for thinking of us.After reviewing your application, we’ve decided to move forward with other candidates for the Software Engineer Intern - Summer 2024 role."], "ENCODE_CAT": [1]}
df = pd.DataFrame(data)

training_set = TriageDataset(df, tokenizer, Config.MAX_LEN)

train_params = {'batch_size': Config.TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

training_loader = DataLoader(training_set, **train_params)

model = BERTClassifier().to(Config.device)

for _, data in enumerate(training_loader, 0):
    ids = data['ids'].to(Config.device, dtype=torch.long)
    mask = data['mask'].to(Config.device, dtype=torch.long)
    token_type_ids = data['token_type_ids'].to(Config.device, dtype=torch.long)
    outputs = model(ids, mask, token_type_ids)
    big_val, big_idx = torch.max(outputs.data, dim=1)
    print(outputs)
    print(big_idx)