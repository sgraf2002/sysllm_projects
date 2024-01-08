import torch
from datasets import load_dataset
import pdb
from transformers import GPT2Tokenizer, GPT2Model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
dataset_repo = "databricks/databricks-dolly-15k"
dataset = load_dataset(dataset_repo, split='train')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.truncation_side='left'


model = GPT2Model.from_pretrained('gpt2', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, force_download=True)

def process_dataset(example):
    return tokenizer([" ".join([example['context'][i], example['instruction'][i], example['response'][i]]) for i in range(len(example['response']))], truncation=True, max_length=1024)

tokenized_dataset = dataset.map(process_dataset, batched=True, remove_columns=dataset.column_names)
pdb.set_trace()

