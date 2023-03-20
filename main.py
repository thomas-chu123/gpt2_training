from transformers import GPT2Tokenizer, GPT2LMHeadModel, TFGPT2LMHeadModel, Trainer, TrainingArguments
import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.kersa.layers import Dense, LSTM, Embedding, Dropout, Bidirectional
# from tensorflow.keras.optimizers import SGD, Adam
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.datasets import imdb, mnist
from transformers import TFAutoModelForCausalLM, AutoTokenizer, AdamWeightDecay, pipeline, create_optimizer
from transformers import DefaultDataCollator
import tensorflow as tf
from datasets import Dataset, DatasetDict, load_dataset
import plotly.express as px
import plotly.io as pio
import pandas as pd
import math
import os

## create a function to handle keras mnist dataset
def start_training():

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pio.renderers.default = 'notebook_connected'

    with tf.device('/GPU:0'):
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = TFAutoModelForCausalLM.from_pretrained("distilgpt2", pad_token_id=tokenizer.eos_token_id)

        data = load_dataset("CShorten/ML-ArXiv-Papers", split='train')
        data = data.train_test_split(shuffle=True, seed=200, test_size=0.2)

        train = data["train"]
        val = data["test"]

        # The tokenization function
        def tokenization(data):
            tokens = tokenizer(data["abstract"], padding="max_length", truncation=True, max_length=300)
            return tokens

        # Apply the tokenizer in batch mode and drop all the columns except the tokenization result
        train_token = train.map(tokenization, batched=True,
                                remove_columns=["title", "abstract", "Unnamed: 0", "Unnamed: 0.1"], num_proc=10)
        val_token = val.map(tokenization, batched=True, remove_columns=["title", "abstract", "Unnamed: 0", "Unnamed: 0.1"],
                            num_proc=10)

        # Create labels as a copy of input_ids
        def create_labels(text):
            text["labels"] = text["input_ids"].copy()
            return text

        # Add the labels column using map()
        lm_train = train_token.map(create_labels, batched=True, num_proc=10)
        lm_val = val_token.map(create_labels, batched=True, num_proc=10)

        train_set = model.prepare_tf_dataset(
            lm_train,
            shuffle=True,
            batch_size=16
        )

        validation_set = model.prepare_tf_dataset(
            lm_val,
            shuffle=False,
            batch_size=16
        )
        # Setting up the learning rate scheduler
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0005,
            decay_steps=500,
            decay_rate=0.95,
            staircase=False)

        # Exponential decay learning rate
        optimizer = AdamWeightDecay(learning_rate=lr_schedule, weight_decay_rate=0.01)
        model.compile(optimizer=optimizer)
        model.summary()

        # This cell is optional
        # from transformers.keras_callbacks import PushToHubCallback

        # model_name = "GPT-2"
        # push_to_hub_model_id = f"{model_name}-finetuned-papers"

        # push_to_hub_callback = PushToHubCallback(
        #    output_dir="./clm_model_save",
        #    tokenizer=tokenizer,
        #    hub_model_id=push_to_hub_model_id,
        #    hub_token="your HF token"
        # )
        # This cell is optional
        from tensorflow.keras.callbacks import TensorBoard

        tensorboard_callback = TensorBoard(log_dir="./tensorboard",
                                           update_freq=1,
                                           histogram_freq=1,
                                           profile_batch="2,10")

        # callbacks = [push_to_hub_callback, tensorboard_callback]
        # Fit with callbacks

        model.fit(train_set, validation_data=validation_set, epochs=1, workers=100, use_multiprocessing=True, callbacks=tensorboard_callback)
        # model.fit(train_set, validation_data=validation_set, epochs=20, workers=9, use_multiprocessing=True)
        eval_loss = model.evaluate(validation_set)
        model.save_pretrained("gpt2_finetuned")
        print(eval_loss)

def data_handling():
    print()

def gpt2_pt_finetune():
    # Load the custom programming language data
    with open('data.dlf', 'r') as f:
        code_text = f.read()

    code_text = code_text.split('\n')

    # Initialize the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    model = model.to('cuda')

    # Fine-tune the model on the custom programming language data
    count = 0
    train_data = []
    for data in code_text:
        train_data.append(tokenizer(data, return_tensors='pt', max_length=250, truncation=True, padding=True).to('cuda'))
        print(count, len(code_text))
        count = count + 1

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
    trainer.train().to('cuda')

    # Generate code completions for a partial code snippet
    partial_code = 'login web page'
    input_ids = tokenizer.encode(partial_code, return_tensors='pt').to('cuda')
    output_ids = model.generate(input_ids, max_length=500, do_sample=True)[0]
    completion = tokenizer.decode(output_ids, skip_special_tokens=True).to('cuda')

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


class ProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, loss={logs['loss']}, val_loss={logs['val_loss']}")


def gpt2_tf_finetune():
    model = TFGPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    with open("data.dlf", "r", encoding="utf-8") as f:
        text = f.read()

    text_list = text.split("\n")

    with tf.device('/GPU:0'):
        count = 0
        input_ids = []
        for text_line in text_list:
            print (count, len(text_list))
            input_ids.append(tokenizer.encode(text_line, return_tensors="tf", max_length=2, truncation=True))
            count = count + 1

        #input_ids = tokenizer(text_list, return_tensors="tf")
        model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="adam")
        model.fit(input_ids, input_ids, epochs=100, callbacks=[ProgressCallback()])

        model.save_pretrained("./acts_model")
        tokenizer.save_pretrained("./acts_model")

        tokenizer = GPT2Tokenizer.from_pretrained("./acts_model")
        model = TFGPT2LMHeadModel.from_pretrained("./acts_model")

        input_text = "login web page and verify the login is successful"
        input_ids = tokenizer.encode(input_text, return_tensors="tf")
        output = model.generate(input_ids, max_length=50, do_sample=True)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(generated_text)

def test_gpt():
    import torch
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_name = "gpt2-xl"
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    input_txt = "I have a pen, I have an "
    input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
    iterations = []
    n_steps = 10
    choices_per_step = 3

    with torch.no_grad():
        for _ in range(n_steps):
            iteration = dict()
            iteration["Input"] = tokenizer.decode(input_ids[0])
            output = model(input_ids)
            # 選最後一個 token 然後過 softmax 後選出機率最大
            next_token_logits = output.logits[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)

            input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)
            iterations.append(iteration)

    print(iterations[-1])

if __name__ == '__main__':
    # gpt2_pt_finetune()
    # gpt2_tf_finetune()
    start_training()
