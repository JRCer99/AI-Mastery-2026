import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from datetime import datetime

def create_sample_data():
    data = {
        'text': [
            "Congratulations! You've won a free prize. Claim now!",
            "Meeting at 3pm tomorrow in conference room",
            "Your account has been suspended. Verify immediately.",
            "Team lunch this Friday at 12:30",
            "URGENT: Update your banking details or lose access",
            "Project deadline extended to next week"
        ],
        'label': [1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = ham
    }
    return pd.DataFrame(data)

def main():
    print("🚀 Fine-tuning BERT for Text Classification")
    print("=" * 60)

    df = create_sample_data()
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

    train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
    test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",
        logging_dir='./logs',
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    print("Training BERT...")
    trainer.train()

    model.save_pretrained("bert_spam_classifier")
    tokenizer.save_pretrained("bert_spam_classifier")
    print("✅ Fine-tuned BERT model saved!")

    test_text = "Congratulations! You've won $1000. Claim now!"
    model.to('cpu')
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    print(f"\nTest: '{test_text}' → {'SPAM' if prediction == 1 else 'HAM'}")

    print(f"\n🎉 Month 6 Project 1 Complete! - {datetime.now().strftime('%B %d, %Y')}")

if __name__ == "__main__":
    main()
