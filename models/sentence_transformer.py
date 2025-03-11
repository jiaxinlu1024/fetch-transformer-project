import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    """
    Implements multi-head self-attention mechanism.
    """
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"

        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # Linear projections for queries, keys, and values
        self.query_proj = nn.Linear(embed_size, embed_size)
        self.key_proj = nn.Linear(embed_size, embed_size)
        self.value_proj = nn.Linear(embed_size, embed_size)
        self.output_proj = nn.Linear(embed_size, embed_size)

    def forward(self, queries, keys, values, mask=None):
        batch_size = queries.shape[0]

        # Compute queries, keys, and values
        Q = self.query_proj(queries)
        K = self.key_proj(keys)
        V = self.value_proj(values)

        # Reshape to multi-head format
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scaling_factor = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scaling_factor

        if mask is not None:
            attention_scores += mask * -10000.0  # Large negative value to mask attention

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Reshape back to original format
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        return self.output_proj(attention_output)

class Encoder(nn.Module):
    """
    Transformer encoder layer with self-attention and feed-forward network.
    """
    def __init__(self, embed_size, num_heads, hidden_dim, dropout=0.1, layer_norm_eps=1e-12):
        super(Encoder, self).__init__()
        self.self_attention = MultiHeadSelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_size, eps=layer_norm_eps)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward network with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class SentenceTransformer(nn.Module):
    """
    Sentence Transformer model with token embeddings, positional embeddings, and multiple encoder layers.
    Supports different output pooling strategies.
    """
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, max_length,
                 pooling="cls", output_mode="sentence", layer_norm_eps=1e-12):
        super(SentenceTransformer, self).__init__()
        self.embed_size = embed_size
        self.pooling = pooling  # "cls" or "mean" for sentence embeddings
        self.output_mode = output_mode  # "sentence" or "token"

        # Token, position, and token type embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_length, embed_size)
        self.token_type_embeddings = nn.Embedding(2, embed_size)

        # Stacked transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            Encoder(embed_size, num_heads, hidden_dim, layer_norm_eps=layer_norm_eps)
            for _ in range(num_layers)
        ])

        self.embed_layer_norm = nn.LayerNorm(embed_size, eps=layer_norm_eps)
        self.embed_dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        batch_size, seq_len = input_ids.shape

        # Compute token, position, and token type embeddings
        token_embed = self.token_embeddings(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        position_embed = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embed = self.token_type_embeddings(token_type_ids)

        # Combine embeddings
        x = token_embed + position_embed + token_type_embed
        x = self.embed_layer_norm(x)
        x = self.embed_dropout(x)

        # Create attention mask if provided
        mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2).to(x.device) if attention_mask is not None else None

        # Pass through transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)

        # Apply output pooling strategy
        if self.output_mode == "sentence":
            if self.pooling == "cls":
                return x[:, 0, :]  # CLS token representation
            elif self.pooling == "mean":
                return x.mean(dim=1)  # Mean pooling across all tokens
        elif self.output_mode == "token":
            return x  # Return all token embeddings
