import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################
# Attention Pooling Module
##############################################
class AttentionPooling(nn.Module):
    def __init__(self, model_dim):
        super(AttentionPooling, self).__init__()
        # Learnable query vector for pooling
        self.pool_query = nn.Parameter(torch.randn(model_dim))
    
    def forward(self, x):
        # x: [batch, seq_len, model_dim]
        # Compute dot-product scores between each time step and the pool query
        scores = torch.matmul(x, self.pool_query)  # [batch, seq_len]
        weights = F.softmax(scores, dim=1)           # [batch, seq_len]
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)  # [batch, model_dim]
        return pooled

##############################################
# Cross Attention Block with Residual Connections
##############################################
class CrossAttentionBlock(nn.Module):
    def __init__(self, model_dim, n_heads, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.ReLU(),
            nn.Linear(model_dim * 4, model_dim)
        )
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, primary, opponent):
        # primary & opponent: [batch, seq_len, model_dim]
        attn_output, _ = self.cross_attn(query=primary, key=opponent, value=opponent)
        # Residual connection for cross attention
        primary = self.norm1(primary + self.dropout1(attn_output))
        # Feed-forward sub-block with residual connection
        ff_output = self.ff(primary)
        primary = self.norm2(primary + self.dropout2(ff_output))
        return primary

##############################################
# Updated Cross Attention Transformer
##############################################
class CrossAttentionTransformer(nn.Module):
    def __init__(self, primary_input_dim, opponent_input_dim, model_dim, n_heads, num_layers, sequence_length, num_cross_blocks=2, dropout=0.1):
        super(CrossAttentionTransformer, self).__init__()
        self.sequence_length = sequence_length
        # Separate embedding layers for primary and opponent inputs
        self.primary_embedding = nn.Linear(primary_input_dim, model_dim)
        self.opponent_embedding = nn.Linear(opponent_input_dim, model_dim)
        
        # Separate transformer encoders for each input branch
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, batch_first=True)
        self.primary_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.opponent_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Stack of cross attention blocks
        self.cross_blocks = nn.ModuleList([CrossAttentionBlock(model_dim, n_heads, dropout=dropout) for _ in range(num_cross_blocks)])
        
        # Attention pooling for each branch
        self.pooling = AttentionPooling(model_dim)
        
        # Final regressor takes concatenated pooled representations
        self.regressor = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 1)
        )
    
    def forward(self, primary_seq, opponent_seq):
        # primary_seq: [batch, seq_len, primary_input_dim]
        # opponent_seq: [batch, seq_len, opponent_input_dim]
        primary_emb = self.primary_embedding(primary_seq)    # [batch, seq_len, model_dim]
        opponent_emb = self.opponent_embedding(opponent_seq)   # [batch, seq_len, model_dim]
        
        primary_encoded = self.primary_encoder(primary_emb)    # [batch, seq_len, model_dim]
        opponent_encoded = self.opponent_encoder(opponent_emb)   # [batch, seq_len, model_dim]
        
        # Apply additional cross attention blocks
        for block in self.cross_blocks:
            primary_encoded = block(primary_encoded, opponent_encoded)
        
        # Use attention pooling to get fixed-size representations
        pooled_primary = self.pooling(primary_encoded)         # [batch, model_dim]
        pooled_opponent = self.pooling(opponent_encoded)       # [batch, model_dim]
        
        combined = torch.cat([pooled_primary, pooled_opponent], dim=1)  # [batch, 2*model_dim]
        output = self.regressor(combined)                      # [batch, 1]
        return output.squeeze(1)

##############################################
# Batter and Pitcher Models Using the Updated Transformer
##############################################
class BatterTransformerModel(nn.Module):
    def __init__(self, batter_input_dim, opp_pitcher_input_dim, model_dim=64, n_heads=4, num_layers=2, sequence_length=3, num_cross_blocks=2):
        super(BatterTransformerModel, self).__init__()
        self.transformer = CrossAttentionTransformer(batter_input_dim, opp_pitcher_input_dim, model_dim, n_heads, num_layers, sequence_length, num_cross_blocks)
    
    def forward(self, batter_seq, opp_pitcher_seq):
        return self.transformer(batter_seq, opp_pitcher_seq)
    
    def load_pretrained_encoder(self, pretrained_model):
        # Load pretrained weights into the primary branch for batter inputs
        self.transformer.primary_embedding.weight.data.copy_(pretrained_model.embedding.weight.data)
        self.transformer.primary_embedding.bias.data.copy_(pretrained_model.embedding.bias.data)
        self.transformer.primary_encoder.load_state_dict(pretrained_model.encoder.state_dict())

class PitcherTransformerModel(nn.Module):
    def __init__(self, pitcher_input_dim, opp_batter_input_dim, model_dim=64, n_heads=4, num_layers=2, sequence_length=3, num_cross_blocks=2, dropout=0.1):
        super(PitcherTransformerModel, self).__init__()
        self.transformer = CrossAttentionTransformer(pitcher_input_dim, opp_batter_input_dim, model_dim, n_heads, num_layers, sequence_length, num_cross_blocks, dropout)
    
    def forward(self, pitcher_seq, opp_batter_seq):
        return self.transformer(pitcher_seq, opp_batter_seq)
    
    def load_pretrained_encoder(self, pretrained_model):
        # Load pretrained weights into the primary branch for pitcher inputs
        self.transformer.primary_embedding.weight.data.copy_(pretrained_model.embedding.weight.data)
        self.transformer.primary_embedding.bias.data.copy_(pretrained_model.embedding.bias.data)
        self.transformer.primary_encoder.load_state_dict(pretrained_model.encoder.state_dict())


##############################################
# Masked Pretraining Transformer Model (Autoencoder)
##############################################
class MaskedPretrainingTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, n_heads, num_layers, sequence_length, dropout=0.0, mask_prob=0.15):
        """
        A masked transformer autoencoder.
        It randomly masks some inputs and tries to reconstruct them.
        
        Args:
            input_dim: Dimensionality of the input features.
            model_dim: Dimension of the transformer model.
            n_heads: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            sequence_length: Length of the input sequence.
            dropout: Dropout rate.
            mask_prob: Probability of masking a token (time step).
        """
        super(MaskedPretrainingTransformer, self).__init__()
        self.sequence_length = sequence_length
        self.mask_prob = mask_prob
        
        # Embedding layer to project inputs into model space
        self.embedding = nn.Linear(input_dim, model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Reconstruction head applied at each time step to reconstruct the input features
        self.reconstruction_head = nn.Linear(model_dim, input_dim)
        
        # Learnable mask token to substitute masked positions (optional)
        self.mask_token = nn.Parameter(torch.randn(input_dim))
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim] input sequence.
            
        Returns:
            reconstructed: [batch, seq_len, input_dim] reconstruction of the sequence.
            mask: [batch, seq_len] Boolean mask indicating which positions were masked.
        """
        # Create a random mask for each time step (masking entire rows)
        # mask: True indicates the token is masked
        mask = torch.rand(x.size(0), x.size(1), device=x.device) < self.mask_prob
        
        # Replace masked positions with the learned mask token.
        # We expand the mask_token to match x's shape.
        mask_token_expanded = self.mask_token.unsqueeze(0).unsqueeze(0).expand_as(x)
        x_masked = x.clone()
        x_masked[mask] = mask_token_expanded[mask]
        
        # Encode the masked input sequence
        embedded = self.embedding(x_masked)      # [batch, seq_len, model_dim]
        encoded = self.encoder(embedded)           # [batch, seq_len, model_dim]
        reconstructed = self.reconstruction_head(encoded)  # [batch, seq_len, input_dim]
        
        return reconstructed, mask
    
##############################################
# Pretraining Transformer Model (Autoencoder)
##############################################
class PretrainingTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, n_heads, num_layers, sequence_length, dropout=0.0):
        """
        A transformer autoencoder that encodes an input sequence and then reconstructs it.
        Updated to include dropout and mirror the encoder settings of the main model.
        """
        super(PretrainingTransformer, self).__init__()
        self.sequence_length = sequence_length
        # Embedding layer to project inputs into model space
        self.embedding = nn.Linear(input_dim, model_dim)
        # Create a stack of transformer encoder layers with dropout and residual connections (built into the layer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Reconstruction head applied at each time step to reconstruct the input
        self.reconstruction_head = nn.Linear(model_dim, input_dim)
    
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        embedded = self.embedding(x)           # [batch, seq_len, model_dim]
        encoded = self.encoder(embedded)         # [batch, seq_len, model_dim]
        reconstructed = self.reconstruction_head(encoded)  # [batch, seq_len, input_dim]
        return reconstructed