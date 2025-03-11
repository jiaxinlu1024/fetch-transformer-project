import torch.nn as nn
from models.sentence_transformer import SentenceTransformer


class MultiTaskSentenceTransformer(SentenceTransformer):
    """
    Multi-task Sentence Transformer supporting both classification and NER tasks.
    """

    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, max_length,
                 num_classes=2, num_entity_labels=17, pooling="cls", layer_norm_eps=1e-12):
        super().__init__(vocab_size=vocab_size, embed_size=embed_size, num_heads=num_heads, hidden_dim=hidden_dim,
                         num_layers=num_layers, max_length=max_length, pooling=pooling, output_mode="token",
                         layer_norm_eps=layer_norm_eps)

        # Output layers for classification and named entity recognition (NER)
        self.classification_head = nn.Linear(embed_size, num_classes)
        self.ner_head = nn.Linear(embed_size, num_entity_labels)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Processes input through the transformer and generates outputs for classification and NER.
        """
        sequence_output = super().forward(input_ids, attention_mask, token_type_ids)

        # Apply pooling strategy to get sentence-level representation
        if self.pooling == "cls":
            cls_output = sequence_output[:, 0, :]  # Use CLS token output
        elif self.pooling == "mean":
            if attention_mask is not None:
                masked_output = sequence_output * attention_mask.unsqueeze(-1)
                cls_output = masked_output.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(
                    -1)  # Mean pooling with masking
            else:
                cls_output = sequence_output.mean(dim=1)  # Simple mean pooling

        # Generate logits for classification and NER
        classification_logits = self.classification_head(cls_output)
        ner_logits = self.ner_head(sequence_output)

        return classification_logits, ner_logits
