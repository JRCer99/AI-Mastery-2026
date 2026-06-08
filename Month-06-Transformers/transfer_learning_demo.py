from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from datetime import datetime

def create_movie_review_data():
    data = {
        'text': [
            "This movie was absolutely fantastic! Best film of the year.",
            "Terrible acting and boring plot. Waste of time.",
            "Great visuals but weak story. Worth watching once.",
            "One of the best movies I've ever seen. Masterpiece!",
            "Horrible script and bad direction. Do not recommend.",
            "Surprisingly good! Better than I expected."
        ],
        'label': [1, 0, 1, 1, 0, 1]  # 1 = positive, 0 = negative
    }
    return pd.DataFrame(data)

def main():
    print("🚀 Transfer Learning Demo - Adapting BERT to Movie Reviews")
    print("=" * 70)

    df = create_movie_review_data()
    train_df, eval_df = train_test_split(df, test_size=0.3, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

    train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
    eval_dataset = Dataset.from_pandas(eval_df).map(tokenize_function, batched=True)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    training_args = TrainingArguments(
        output_dir="./transfer_results",
        num_train_epochs=4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    print("Fine-tuning BERT on new domain (movie reviews)...")
    trainer.train()

    model.save_pretrained("bert_movie_review_classifier")
    tokenizer.save_pretrained("bert_movie_review_classifier")
    print("✅ Transfer-learned model saved!")

    test_text = "The acting was phenomenal and the story kept me engaged throughout!"
    model.to('cpu')
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    print(f"\nTest: '{test_text}' → {'Positive' if prediction == 1 else 'Negative'}")

    print(f"\n🎉 Month 6 Project 2 Complete! - {datetime.now().strftime('%B %d, %Y')}")

if __name__ == "__main__":
    main()
