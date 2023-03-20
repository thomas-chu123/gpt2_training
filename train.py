from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from transformers import Trainer, TrainingArguments
import torch

def main():
    torch.cuda.empty_cache()

    train_test_dataset = (load_dataset('json', data_files='source/train.json',split='train')).train_test_split(test_size=0.2)
    print(train_test_dataset['train'][0])

    model_name = "distilgpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = TFGPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def case_name_tokenize(sets):
        return tokenizer(sets["case_name"], sets["case_script"],padding="max_length", max_length=256, truncation=True)
    case_name_ids = train_test_dataset.map(case_name_tokenize, batched=True)

    # def case_script_tokenize(sets):
    #    return tokenizer(sets["case_script"], padding="max_length", truncation=True)
    # case_script_ids = train_test_dataset.map(case_script_tokenize, batched=True)
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_steps=100,
        save_total_limit=2,
        # prediction_loss_only=True,
        # evaluation_strategy='steps',
    )

    # training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=case_name_ids["train"],
        eval_dataset=case_name_ids["test"],
        tokenizer=tokenizer,
    )
    trainer.train()

    # model.compile(optimizer='adam', loss=model.compute_loss)
    # model.fit(case_name_ids, case_script_ids, epochs=5)



if __name__ == '__main__':
    main()
