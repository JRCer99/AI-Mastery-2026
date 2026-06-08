from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import torch
from datetime import datetime

def create_domain_data():
    data = {
        "text": [
            "Q: How do I break into AI engineering? A: Start with Python, then learn PyTorch, build projects, and fine-tune models.",
            "Q: Best way to learn transformers? A: Understand attention mechanism first, then implement BERT fine-tuning.",
            "Q: Should I use Claude or Grok? A: Both are excellent. Use Grok for real-time knowledge and Claude for long context.",
            "Q: How to build a strong GitHub portfolio? A: Document every project clearly, include READMEs, and show end-to-end pipelines.",
            "Q: Tips for staying motivated in coding? A: Build projects you care about and track your monthly progress."
        ]
    }
    return Dataset.from_dict(data)

def main():
    print("🚀 Fine-tuning a Domain-Specific LLM (AI Career Advisor)")
    print("=" * 70)

    model_name = "gpt2"  # lightweight — swap for microsoft/Phi-2 if GPU available
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], lora_dropout=0.05)
    model = get_peft_model(model, lora_config)

    dataset = create_domain_data()

    def tokenize_function(examples):
        tokens = tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")
        tokens["labels"] = tokens["input_ids"].copy()  # causal LM: labels = input_ids
        return tokens

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir="./phi_domain_results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_strategy="epoch",
        logging_steps=5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Fine-tuning on domain data...")
    trainer.train()

    model.save_pretrained("domain_ai_advisor")
    tokenizer.save_pretrained("domain_ai_advisor")
    print("✅ Domain-specific LLM saved!")

    model.eval()
    prompt = "Q: What's the best way to learn AI in 2026? A:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
    print("\nModel Response:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    print(f"\n🎉 Month 7 Project 1 Complete! - {datetime.now().strftime('%B %d, %Y')}")

if __name__ == "__main__":
    main()
