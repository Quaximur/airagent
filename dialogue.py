from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 128

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map=None
).to(device)

model.eval()

messages = [
    {
        "role": "system",
        "content": "You are a helpful, concise AI assistant."
    }
]

print("🤖 Chat with Qwen3-0.6B")
print("Type 'exit' to quit.")
print("-" * 60)

while True:
    user_input = input("👤 You: ").strip()

    if user_input.lower() in {"exit", "quit", "выход"}:
        print("👋 Goodbye!")
        break

    if not user_input:
        continue

    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    assistant_reply = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    print(f"🤖 Bot: {assistant_reply}")
    print("-" * 60)

    messages.append(
        {
            "role": "assistant",
            "content": assistant_reply
        }
    )

    if len(messages) > 21:
        messages = [messages[0]] + messages[-20:]