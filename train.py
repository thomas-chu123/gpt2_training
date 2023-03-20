from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import torch

def main():
    train_test_dataset = (load_dataset('json', data_files='source/train.json',split='train')).train_test_split(test_size=0.2)
    print(train_test_dataset['train'][0])

    model_name = "distilgpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TFGPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2").to(device)

    #

    input_ids = tokenizer.map()# model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    model.fit(input_ids, input_ids, epochs=5)



if __name__ == '__main__':
    main()
