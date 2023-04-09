from transformers import AutoModelForCausalLM, AutoTokenizer

prompt = "login gui and check the login is successful"
default_tokenizer = "distilgpt2"
# custom_data = "E:\\Python_Program\\gpt2_training\\acts_model"
custom_model = "tp_model"

tokenizer = AutoTokenizer.from_pretrained(custom_model)
inputs = tokenizer(prompt, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(custom_model)
outputs = model.generate(**inputs, max_length=256, do_sample=True, top_k=50, top_p=0.95, temperature=0.9, num_return_sequences=1)
eval = model.eval()
text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
for line in text:
    print(line)

# print(eval)