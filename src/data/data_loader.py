import os
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset


class cusDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label)

        }

def load_and_prepare_data(dataset_name, tokenizer_name, batch_size=16, max_length=512):
    # Loading dataset
    dataset = load_dataset(dataset_name)
    
    # Loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Preparing data splits
    train_texts = dataset["train"]["text"] if "text" in dataset["train"].features else dataset["train"]["document"]
    train_labels = dataset["train"]["label"] if "label" in dataset["train"].features else [0] * len(train_texts)
    
    val_texts = dataset["validation"]["text"] if "validation" in dataset and "text" in dataset["validation"].features else dataset["train"]["document"][:100]
    val_labels = dataset["validation"]["label"] if "validation" in dataset and "label" in dataset["validation"].features else [0] * len(val_texts)
    
    test_texts = dataset["test"]["text"] if "test" in dataset and "text" in dataset["test"].features else dataset["train"]["document"][:100]
    test_labels = dataset["test"]["label"] if "test" in dataset and "label" in dataset["test"].features else [0] * len(test_texts)
    
    # Creating datasets
    train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)
    
    # Creating dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_dataloader, val_dataloader, test_dataloader

# Data augmentation function
def augment_data(texts, labels, augmentation_factor=2):
    import random
    import nltk
    nltk.download('wordnet')
    from nltk.corpus import wordnet
    
    augmented_texts = []
    augmented_labels = []
    
    for text, label in zip(texts, labels):
        augmented_texts.append(text)
        augmented_labels.append(label)
        
        words = text.split()
        
        for _ in range(augmentation_factor - 1):
            new_words = words.copy()
            
            # Replacing random words with synonyms
            for i in range(len(new_words)):
                if random.random() < 0.2:  # 20% chance to replace a word
                    synonyms = []
                    for syn in wordnet.synsets(new_words[i]):
                        for lemma in syn.lemmas():
                            synonyms.append(lemma.name())
                    
                    if synonyms:
                        new_words[i] = random.choice(synonyms)
            
            augmented_text = ' '.join(new_words)
            augmented_texts.append(augmented_text)
            augmented_labels.append(label)
    
    return augmented_texts, augmented_labels