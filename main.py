from transformers import GPT2Tokenizer, GPT2LMHeadModel, TFGPT2LMHeadModel, Trainer, TrainingArguments

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from Dataset import Dataset





def gpt2_pt_finetune():
    # Load the custom programming language data
    with open('data.dlf', 'r') as f:
        code_text = f.read()

    code_text = code_text.split('\n')

    # Initialize the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Fine-tune the model on the custom programming language data
    train_data = tokenizer(code_text, return_tensors='pt', truncation=True, padding=True)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy='steps',
        eval_steps=500,
        save_total_limit=5,
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data
    )
    trainer.train()

    # Generate code completions for a partial code snippet
    partial_code = 'login web page'
    input_ids = tokenizer.encode(partial_code, return_tensors='pt')
    output_ids = model.generate(input_ids, max_length=50, do_sample=True)[0]
    completion = tokenizer.decode(output_ids, skip_special_tokens=True)

    print('Code completion:', completion)


# create transformer function to convert the data into the format that the model expects
def convert_to_transformer_inputs(tokenizer, df, text_column, max_len=512):
    """Convert dataframe column to lists of input ids, attention masks and labels"""
    input_ids = []
    attention_masks = []
    labels = []
    for i in range(len(df)):
        encoded_dict = tokenizer.encode_plus(
            df[text_column][i],
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(encoded_dict['input_ids'])
    return input_ids, attention_masks, labels


def gpt2_tf_finetune():
    model = TFGPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    with open("data.dlf", "r", encoding="utf-8") as f:
        text = f.read()

    text_list = text.split("\n")

    input_ids = []
    for text_line in text_list:
        input_ids.append(tokenizer.encode(text_line, return_tensors="tf", max_length=10, truncation=True))

    #input_ids = tokenizer(text_list, return_tensors="tf")
    model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="adam")
    model.fit(input_ids, input_ids, epochs=5)

    model.save_pretrained("./acts_model")
    tokenizer.save_pretrained("./acts_model")

    tokenizer = GPT2Tokenizer.from_pretrained("./acts_model")
    model = TFGPT2LMHeadModel.from_pretrained("./acts_model")

    input_text = "login web page and verify the login is successful"
    input_ids = tokenizer.encode(input_text, return_tensors="tf")
    output = model.generate(input_ids, max_length=50, do_sample=True)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)


if __name__ == '__main__':
    # gpt2_pt_finetune()
    gpt2_tf_finetune()
