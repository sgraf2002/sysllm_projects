import torch
from datasets import load_dataset
import pdb
from transformers import GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForCausalLM

from pynvml import *

   

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(tokens_per_second, avg_training_time_epoch, training_loss_per_epoch):
    for epoch, loss in enumerate(training_loss_per_epoch):
        print(f"Train loss epoch{epoch}: {loss['loss']}")
    print(f"Tokens/second: {tokens_per_second}")
    print(f"Runtime/epoch: {avg_training_time_epoch}")
    print_gpu_utilization()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 1024
print(device)
dataset_repo = "databricks/databricks-dolly-15k"
dataset = load_dataset(dataset_repo, split='train')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.truncation_side='left'

model = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

def process_dataset(example):
    processed_example = {}
    processed_example["input_ids"] = tokenizer([" ".join([example['context'][i], example['instruction'][i], example['response'][i]]) for i in range(len(example['response']))], truncation=True, max_length=max_length, padding=True)["input_ids"]
    processed_example["labels"] = processed_example["input_ids"].copy()
    return processed_example

tokenizer.pad_token = tokenizer.eos_token
tokenized_dataset = dataset.map(process_dataset, batched=True, remove_columns=dataset.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.005)["test"]
num_tokens_dataset = len(tokenized_dataset) * max_length



training_args = TrainingArguments(
    per_device_train_batch_size = 1,
    per_device_eval_batch_size = 1,
    gradient_accumulation_steps = 16,
    num_train_epochs= 2,
    output_dir= f"../one_gpu_finetune",
    evaluation_strategy="no",
    logging_strategy="epoch",
    learning_rate=2e-3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    data_collator=data_collator
)
result = trainer.train()
training_runtime = result.metrics['train_runtime']
tokens_per_second = num_tokens_dataset * 2 / training_runtime
avg_training_time_epoch = training_runtime / 2
training_loss_per_epoch = trainer.state.log_history[:-1]


print_summary(tokens_per_second, avg_training_time_epoch, training_loss_per_epoch)


