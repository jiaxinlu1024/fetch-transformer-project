from models.sentence_transformer import SentenceTransformer
from models.multi_task_sentence_transformer import MultiTaskSentenceTransformer


def initialize_custom_model(pretrained_model, model_type="sentence", pooling="cls", output_mode="sentence",
                            num_classes=2, num_entity_labels=17):
    """
    Initialize a custom model (either SentenceTransformer or MultiTaskSentenceTransformer) and copy weights 
    from a pretrained BERT model.
    """

    # **Function to Copy Weights from Pretrained BERT Model**
    def copy_pretrained_weights(custom_model, pretrained_model):
        """Copy weights from a pretrained BERT model to the custom SentenceTransformer or MultiTaskSentenceTransformer."""
        # Copy embedding weights
        custom_model.token_embeddings.weight.data.copy_(pretrained_model.embeddings.word_embeddings.weight.data)
        custom_model.position_embeddings.weight.data.copy_(pretrained_model.embeddings.position_embeddings.weight.data)
        custom_model.token_type_embeddings.weight.data.copy_(
            pretrained_model.embeddings.token_type_embeddings.weight.data)
        custom_model.embed_layer_norm.weight.data.copy_(pretrained_model.embeddings.LayerNorm.weight.data)
        custom_model.embed_layer_norm.bias.data.copy_(pretrained_model.embeddings.LayerNorm.bias.data)

        # Copy weights for each encoder layer
        for custom_layer, pretrained_layer in zip(custom_model.encoder_layers, pretrained_model.encoder.layer):
            # Self-attention projections
            custom_layer.self_attention.query_proj.weight.data.copy_(pretrained_layer.attention.self.query.weight.data)
            custom_layer.self_attention.query_proj.bias.data.copy_(pretrained_layer.attention.self.query.bias.data)
            custom_layer.self_attention.key_proj.weight.data.copy_(pretrained_layer.attention.self.key.weight.data)
            custom_layer.self_attention.key_proj.bias.data.copy_(pretrained_layer.attention.self.key.bias.data)
            custom_layer.self_attention.value_proj.weight.data.copy_(pretrained_layer.attention.self.value.weight.data)
            custom_layer.self_attention.value_proj.bias.data.copy_(pretrained_layer.attention.self.value.bias.data)
            custom_layer.self_attention.output_proj.weight.data.copy_(
                pretrained_layer.attention.output.dense.weight.data)
            custom_layer.self_attention.output_proj.bias.data.copy_(pretrained_layer.attention.output.dense.bias.data)

            # Feed-forward network
            custom_layer.feed_forward[0].weight.data.copy_(pretrained_layer.intermediate.dense.weight.data)
            custom_layer.feed_forward[0].bias.data.copy_(pretrained_layer.intermediate.dense.bias.data)
            custom_layer.feed_forward[2].weight.data.copy_(pretrained_layer.output.dense.weight.data)
            custom_layer.feed_forward[2].bias.data.copy_(pretrained_layer.output.dense.bias.data)

            # Layer normalization
            custom_layer.norm1.weight.data.copy_(pretrained_layer.attention.output.LayerNorm.weight.data)
            custom_layer.norm1.bias.data.copy_(pretrained_layer.attention.output.LayerNorm.bias.data)
            custom_layer.norm2.weight.data.copy_(pretrained_layer.output.LayerNorm.weight.data)
            custom_layer.norm2.bias.data.copy_(pretrained_layer.output.LayerNorm.bias.data)
        print("Weights successfully copied from pretrained BERT to custom model!")

    # **Extract Hyperparameters from Pretrained Model**
    config = pretrained_model.config
    vocab_size = config.vocab_size
    embed_size = config.hidden_size
    num_heads = config.num_attention_heads
    hidden_dim = config.intermediate_size
    num_layers = config.num_hidden_layers
    max_length = config.max_position_embeddings
    layer_norm_eps = config.layer_norm_eps  # Use BERT's epsilon

    print(f"Vocab Size: {vocab_size}")
    print(f"Embedding Size: {embed_size}")
    print(f"Number of Heads: {num_heads}")
    print(f"Hidden Dimension: {hidden_dim}")
    print(f"Number of Layers: {num_layers}")
    print(f"Max Position Embeddings: {max_length}")
    print(f"Layer Norm Epsilon: {layer_norm_eps}")

    # **Initialize Custom Model Based on Model Type**
    if model_type == "sentence":
        custom_model = SentenceTransformer(
            vocab_size=vocab_size,
            embed_size=embed_size,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_length=max_length,
            pooling=pooling,
            output_mode=output_mode,
            layer_norm_eps=layer_norm_eps
        )
    elif model_type == "multitask":
        custom_model = MultiTaskSentenceTransformer(
            vocab_size=vocab_size,
            embed_size=embed_size,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_length=max_length,
            num_classes=num_classes,
            num_entity_labels=num_entity_labels,
            pooling=pooling,
            layer_norm_eps=layer_norm_eps
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'sentence' or 'multitask'.")

    # **Copy Weights from Pretrained Model**
    copy_pretrained_weights(custom_model, pretrained_model)

    print("\nCustom model initialized and weights copied successfully.")

    return custom_model
