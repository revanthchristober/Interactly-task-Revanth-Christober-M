from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load dataset
def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128,
    )

# Load data collator
def load_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

# Fine-tune the model
def fine_tune():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Load dataset
    train_dataset = load_dataset("/workspaces/indexify-image-object-detection-reverse-search/Interactly_task_Profile_Matching/data/fine_tuning_data.txt", tokenizer)
    data_collator = load_data_collator(tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="/workspaces/indexify-image-object-detection-reverse-search/Interactly_task_Profile_Matching/models/fine_tuned_model",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Fine-tune model
    trainer.train()
    trainer.save_model("/workspaces/indexify-image-object-detection-reverse-search/Interactly_task_Profile_Matching/models/fine_tuned_model")
    tokenizer.save_pretrained("/workspaces/indexify-image-object-detection-reverse-search/Interactly_task_Profile_Matching/models/fine_tuned_model")

if __name__ == "__main__":
    fine_tune()
