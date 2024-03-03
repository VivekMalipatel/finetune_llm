import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer

from torch import cuda
device = 'cuda:5' if cuda.is_available() else 'cpu'


