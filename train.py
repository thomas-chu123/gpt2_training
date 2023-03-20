from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import torch

def main():
    train_test_dataset = (load_dataset('json', data_files='source/train.json',split='train')).train_test_split(test_size=0.2)
    print(train_test_dataset['train'][0])

    model_name = "distilgpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TFGPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def case_name_tokenize(sets):
        return tokenizer(sets["case_name"], padding="max_length", truncation=True)
    case_name_ids = train_test_dataset.map(case_name_tokenize, batched=True)

    def case_script_tokenize(sets):
        return tokenizer(sets["case_script"], padding="max_length", truncation=True)
    case_script_ids = train_test_dataset.map(case_script_tokenize, batched=True)

    model.fit(case_name_ids, case_script_ids, epochs=5)



if __name__ == '__main__':
    main()
