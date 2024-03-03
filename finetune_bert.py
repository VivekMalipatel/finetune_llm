import pandas as pd
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer
import evaluate
from datasets import Dataset, DatasetDict
import numpy as np

from torch import cuda
device = 'cuda:5' if cuda.is_available() else 'cpu'

model_checkpoint = 'bert-base-cased'
file_path ='preprocessed_emails.csv'
output = 'bert_finetuned'

df = pd.read_csv(file_path)
df = df[['Body', 'Label']]

encode_dict = {'Rejected': 0, 'Applied': 1, 'Irrelevant': 2, 'Accepted' : 3}

def encode_cat(x):
    return encode_dict[x]

df['Label'] = df['Label'].apply(lambda x: encode_cat(x))
df['Body'] = df['Body'].astype(str)

train_size = 0.8
train_dataset=df.sample(frac=train_size,random_state=200)
test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

train_dataset = Dataset.from_pandas(train_dataset)
test_dataset = Dataset.from_pandas(test_dataset)

def tokenize_function(examples):
    return tokenizer(examples["Body"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

print(train_dataset)
print(test_dataset)

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=4)
training_args = TrainingArguments(output_dir=output)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(output_dir=output, evaluation_strategy="epoch")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
