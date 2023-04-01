from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from transformers import DefaultDataCollator
import torch

def main():
    torch.cuda.empty_cache()
    train_test_dataset = (load_dataset('json', data_files='source/train.json',split='train')).train_test_split(test_size=0.2)
    print(train_test_dataset['train'][0])

    model_name = "distilgpt2"
    # model_name = "bert-base-uncased"
    # model_name = "distilbert-base-uncased"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '[SEP]'})

    def case_tokenize(sets):
        return tokenizer(sets["context"],
                         max_length=128,
                         padding=True,
                         truncation=True,
                         # return_overflowing_tokens=True,
                         # return_offsets_mapping=True,
                         )
    train_ids = train_test_dataset['train'].map(case_tokenize, batched=True)
    test_ids = train_test_dataset['test'].map(case_tokenize, batched=True)

    print(tokenizer.decode(train_ids["input_ids"][0]))
    # print(train_ids['input_ids'][0])
    # print(train_ids.keys())

    data_collator = DefaultDataCollator()

    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=1,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_steps=100,
        save_total_limit=2,
        weight_decay=0.1,
        # prediction_loss_only=True,
        evaluation_strategy='epoch',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ids,
        eval_dataset=test_ids,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.train()
    trainer.save_model("acts_model")


if __name__ == '__main__':
    main()
