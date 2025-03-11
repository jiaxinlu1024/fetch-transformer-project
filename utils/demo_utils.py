import torch
import torch.nn.functional as F
import random
from utils.data_utils import prepare_classification_data, prepare_ner_data
from utils.train_utils import set_requires_grad


def demo_emb_predicting(custom_model, pretrained_model, tokenizer, sentences):
    """
    Demonstrates embedding extraction from both a custom model and a pretrained BERT model.
    Compares CLS, mean, and token embeddings using cosine similarity.
    """
    tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        # Extract CLS embeddings
        custom_model.output_mode = "sentence"
        custom_model.pooling = "cls"
        custom_cls_embeddings = custom_model(tokens["input_ids"], tokens["attention_mask"], tokens["token_type_ids"])
        bert_outputs = pretrained_model(**tokens)
        bert_cls_embeddings = bert_outputs.last_hidden_state[:, 0, :]

        # Extract mean embeddings
        custom_model.pooling = "mean"
        custom_mean_embeddings = custom_model(tokens["input_ids"], tokens["attention_mask"], tokens["token_type_ids"])
        bert_token_embeddings = bert_outputs.last_hidden_state
        bert_mean_embeddings = bert_token_embeddings.mean(dim=1)

        # Extract token embeddings (full sequence)
        custom_model.output_mode = "token"
        custom_token_embeddings = custom_model(tokens["input_ids"], tokens["attention_mask"], tokens["token_type_ids"])
        bert_token_embeddings = bert_outputs.last_hidden_state

    # Normalize embeddings for cosine similarity comparison
    custom_cls_embeddings = F.normalize(custom_cls_embeddings, p=2, dim=-1)
    custom_mean_embeddings = F.normalize(custom_mean_embeddings, p=2, dim=-1)
    bert_cls_embeddings = F.normalize(bert_cls_embeddings, p=2, dim=-1)
    bert_mean_embeddings = F.normalize(bert_mean_embeddings, p=2, dim=-1)

    def compute_similarity(emb1, emb2):
        """Computes cosine similarity between two embeddings."""
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

    # Compute similarities for CLS and mean embeddings
    cosine_sim_custom_cls_1_2 = compute_similarity(custom_cls_embeddings[0], custom_cls_embeddings[1])
    cosine_sim_custom_cls_2_3 = compute_similarity(custom_cls_embeddings[1], custom_cls_embeddings[2])
    cosine_sim_custom_mean_1_2 = compute_similarity(custom_mean_embeddings[0], custom_mean_embeddings[1])
    cosine_sim_custom_mean_2_3 = compute_similarity(custom_mean_embeddings[1], custom_mean_embeddings[2])

    cosine_sim_bert_cls_1_2 = compute_similarity(bert_cls_embeddings[0], bert_cls_embeddings[1])
    cosine_sim_bert_cls_2_3 = compute_similarity(bert_cls_embeddings[1], bert_cls_embeddings[2])
    cosine_sim_bert_mean_1_2 = compute_similarity(bert_mean_embeddings[0], bert_mean_embeddings[1])
    cosine_sim_bert_mean_2_3 = compute_similarity(bert_mean_embeddings[1], bert_mean_embeddings[2])

    # Display embedding demonstrations
    print("\n **Embedding Demonstrations**")

    # Display CLS embeddings
    print("\n--- CLS Embeddings ---")
    for i, sentence in enumerate(sentences):
        print(f"Sentence {i + 1}: \"{sentence}\"")
        print(f"Custom Model CLS Embedding: {custom_cls_embeddings[i].tolist()[:5]} ...")  # Print first 5 dimensions
        print(f"Pretrained BERT CLS Embedding: {bert_cls_embeddings[i].tolist()[:5]} ...\n")

    # Display mean embeddings
    print("\n--- Mean Pooling Embeddings ---")
    for i, sentence in enumerate(sentences):
        print(f"Sentence {i + 1}: \"{sentence}\"")
        print(f"Custom Model Mean Embedding: {custom_mean_embeddings[i].tolist()[:5]} ...")
        print(f"Pretrained BERT Mean Embedding: {bert_mean_embeddings[i].tolist()[:5]} ...\n")

    # Display cosine similarity results
    print("\n **Cosine Similarity Results**")
    print("\n--- Custom Model ---")
    print(f"Custom Model CLS Similarity (Sentence 1 & 2): {cosine_sim_custom_cls_1_2:.4f}")
    print(f"Custom Model CLS Similarity (Sentence 2 & 3): {cosine_sim_custom_cls_2_3:.4f}")
    print(f"Custom Model Mean Similarity (Sentence 1 & 2): {cosine_sim_custom_mean_1_2:.4f}")
    print(f"Custom Model Mean Similarity (Sentence 2 & 3): {cosine_sim_custom_mean_2_3:.4f}")

    print("\n--- Pretrained BERT ---")
    print(f"Pretrained BERT CLS Similarity (Sentence 1 & 2): {cosine_sim_bert_cls_1_2:.4f}")
    print(f"Pretrained BERT CLS Similarity (Sentence 2 & 3): {cosine_sim_bert_cls_2_3:.4f}")
    print(f"Pretrained BERT Mean Similarity (Sentence 1 & 2): {cosine_sim_bert_mean_1_2:.4f}")
    print(f"Pretrained BERT Mean Similarity (Sentence 2 & 3): {cosine_sim_bert_mean_2_3:.4f}")

    # Display token embeddings
    print("\n--- Token (Full Sequence) Embeddings ---")
    print(f"Custom Model Token Embeddings Shape: {custom_token_embeddings.shape}")
    print(f"Pretrained BERT Token Embeddings Shape: {bert_token_embeddings.shape}")
    print(
        f"\nCustom Model Token Embedding (First 5 Tokens): {custom_token_embeddings[:, :5, :]} ...")
    print(
        f"Pretrained BERT Token Embedding (First 5 Tokens): {bert_token_embeddings[:, :5, :]} ...")




def demo_data_loading(classification_file, ner_file, sample_ratio=1.0):
    """
    Demonstrates loading of classification and NER datasets with optional sampling.
    Allows loading only a fraction of the dataset for quick testing.

    Args:
        classification_file (str): Path to classification dataset file.
        ner_file (str): Path to NER dataset file.
        sample_ratio (float): Fraction of data to sample (0.0 to 1.0).

    Returns:
        Tuple: Train/test DataLoaders for classification and NER tasks.
    """
    assert 0.0 < sample_ratio <= 1.0, "sample_ratio must be between 0 and 1"

    # Load full datasets
    classification_train_loader, classification_test_loader, classification_train_size, classification_test_size = prepare_classification_data(
        classification_file)
    ner_train_loader, ner_test_loader, ner_train_size, ner_test_size = prepare_ner_data(ner_file)

    def sample_dataloader(dataloader, sample_ratio):
        """Randomly samples a fraction of a DataLoader's dataset."""
        dataset = dataloader.dataset
        sampled_size = max(1, int(len(dataset) * sample_ratio))  # Ensure at least one sample
        sampled_indices = random.sample(range(len(dataset)), sampled_size)
        sampled_dataset = torch.utils.data.Subset(dataset, sampled_indices)
        return torch.utils.data.DataLoader(sampled_dataset, batch_size=dataloader.batch_size, shuffle=True,
                                           collate_fn=dataloader.collate_fn)

    # Apply sampling
    if sample_ratio < 1.0:
        classification_train_loader = sample_dataloader(classification_train_loader, sample_ratio)
        classification_test_loader = sample_dataloader(classification_test_loader, sample_ratio)
        ner_train_loader = sample_dataloader(ner_train_loader, sample_ratio)
        ner_test_loader = sample_dataloader(ner_test_loader, sample_ratio)

    # Print dataset sizes
    print(
        f"Classification Train Size: {len(classification_train_loader.dataset)}, Test Size: {len(classification_test_loader.dataset)}")
    print(f"NER Train Size: {len(ner_train_loader.dataset)}, Test Size: {len(ner_test_loader.dataset)}")

    # Display a sample batch for classification
    for batch_inputs, batch_labels in classification_train_loader:
        print("\n Classification Batch Sample:")
        print("Input IDs shape:", batch_inputs["input_ids"].shape)
        print("Labels shape:", batch_labels.shape)
        break  # Show only the first batch

    # Display a sample batch for NER
    for batch_inputs, batch_labels in ner_train_loader:
        print("\n NER Batch Sample:")
        print("Input IDs shape:", batch_inputs["input_ids"].shape)
        print("Labels shape:", batch_labels.shape)
        break  # Show only the first batch

    return classification_train_loader, classification_test_loader, ner_train_loader, ner_test_loader


def demo_unfreeze_strategy(model):
    """
    Demonstrates different unfreeze strategies and prints the number of trainable parameters.
    """
    strategies = ['heads_only', 'all_adaptive', 'output_layers', 'all']

    for strategy in strategies:
        print("\n" + "=" * 50)
        print(f"Applying Unfreeze Strategy: {strategy}")
        print("=" * 50)

        # Apply the unfreeze strategy
        set_requires_grad(model, strategy)

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        print(f" Trainable Parameters: {trainable_params:,} / {total_params:,}")


def print_task_title(task_name):
    """Prints a clear and formatted title for each task in the console."""
    title = f" {task_name} TASK STARTS "
    border = "=" * len(title)
    print(f"\n{border}\n{title}\n{border}\n")
