from airllm import AutoModel
import torch

MAX_LENGTH = 256
MODEL_NAME = "Qwen/Qwen3-8B"
model = AutoModel.from_pretrained(MODEL_NAME)

print(f"🤖 Чат {MODEL_NAME}")
print("-" * 50)

while True:
    user_input = input("👤 Вы: ")

    if user_input.lower() in ['exit', 'quit', 'выход']:
        print("👋 До свидания!")
        break

    if not user_input.strip():
        continue

    inputs = model.tokenizer(
        user_input,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_attention_mask=True
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_ids = inputs['input_ids'].cuda()
    attention_mask = inputs['attention_mask'].to(device)

    generation_output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=200,
        use_cache=True,
        return_dict_in_generate=True,
        do_sample=True,
        temperature=0.7
    )

    output = model.tokenizer.decode(generation_output.sequences[0])

    if output.startswith(user_input):
        output = output[len(user_input):].strip()

    print(f"🤖 Бот: {output}")
    print("-" * 50)