from torch.utils.data import Dataset
import torch
class MultiTaskDataset(Dataset):
    """
    Custom dataset for multi-task learning, supporting classification and NER tasks.
    """
    def __init__(self, task, data, tokenizer, label_map=None, max_length=512):
        self.task = task
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Tokenizes input and returns processed tensors for the specified task."""
        if self.task == "classification":
            text = self.data[idx]["sentence"]
            label = 0 if self.data[idx]["label"] == "negative" else 1
            inputs = self.tokenizer(text, truncation=True, padding="max_length",
                                    max_length=self.max_length, return_tensors="pt")
            return {k: v.squeeze(0) for k, v in inputs.items()}, torch.tensor(label, dtype=torch.long)

        elif self.task == "ner":
            words = self.data[idx]["words"]
            labels = self.data[idx]["labels"]
            inputs = self.tokenizer(words, is_split_into_words=True, truncation=True,
                                    padding="max_length", max_length=self.max_length, return_tensors="pt")
            word_ids = inputs.word_ids(batch_index=0) if hasattr(inputs, "word_ids") else None

            # Initialize all label positions as ignored tokens
            label_ids = [-100] * len(inputs["input_ids"][0])

            if word_ids:
                for i, word_id in enumerate(word_ids):
                    if word_id is not None and word_id < len(labels):
                        if i == 0 or word_ids[i - 1] != word_id:
                            label_ids[i] = self.label_map.get(labels[word_id], -100) # Assign B-label
                        else:
                            label_ids[i] = self.label_map.get(f"I-{labels[word_id][2:]}", -100)  # Assign I-label

            return {k: v.squeeze(0) for k, v in inputs.items()}, torch.tensor(label_ids, dtype=torch.long)

        else:
            raise ValueError(f"Unsupported task: {self.task}. Use 'classification' or 'ner'.")
