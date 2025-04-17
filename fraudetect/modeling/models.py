import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from torchtune.modules import RotaryPositionalEmbeddings
from skorch import NeuralNetClassifier
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNet
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape (1, max_len, d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

class FraudTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=4, num_layers=2, dropout=0.1, num_classes=2):
        """
        input_dim: number of features per transaction
        d_model: embedding dimension
        n_heads: number of attention heads
        num_layers: number of TransformerEncoder layers
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True  # input shape: (batch, seq, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x: tensor of shape (batch_size, seq_len, input_dim)
        """
        # project input features to d_model
        x = self.input_proj(x)                      # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)                     # add positional encodings
        x = self.transformer_encoder(x)             # (batch_size, seq_len, d_model)
        x = x.mean(dim=1)                           # mean pooling over sequence
        x = self.dropout(x)
        x = self.classifier(x)                      # (batch_size, num_classes)
        return x

class FraudGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.1, num_classes=2):
        """
        GRU-based sequence classifier.
        - input_dim: number of features per transaction
        - hidden_dim: GRU hidden size
        - num_layers: number of stacked GRU layers
        - dropout: dropout rate between GRU layers & before classifier
        - num_classes: number of output classes
        """
        super().__init__()
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        # hidden_dim * 2 for bidirectional output
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        # GRU returns: output (batch, seq_len, hidden*2), hidden state
        out, _ = self.gru(x)
        # Pool across sequence dimension
        out = out.mean(dim=1)  # shape: (batch_size, hidden_dim*2)
        out = self.dropout(out)
        return self.classifier(out)


class FraudLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.1, num_classes=2):
        """
        LSTM-based sequence classifier.
        - input_dim: number of features per transaction
        - hidden_dim: LSTM hidden size
        - num_layers: number of stacked LSTM layers
        - dropout: dropout rate between LSTM layers & before classifier
        - num_classes: number of output classes
        """
        super().__init__()
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        # hidden_dim * 2 for bidirectional output
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        # LSTM returns: output (batch, seq_len, hidden*2), (h_n, c_n)
        out, _ = self.lstm(x)
        # Pool across sequence dimension
        out = out.mean(dim=1)  # shape: (batch_size, hidden_dim*2)
        out = self.dropout(out)
        return self.classifier(out)


class FraudTransformerWithRoPE(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dropout=0.1,
        num_classes=2,
        max_seq_len=500,
        rope_base=10000
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Project input features to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Rotary positional embeddings per head
        self.rope = RotaryPositionalEmbeddings(
            dim=self.head_dim,              # per-head dim
            max_seq_len=max_seq_len,
            base=rope_base
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        """
        # 1) Project to model dimension
        x = self.input_proj(x)  # (b, s, d_model)

        # 2) Apply Rotary Positional Embedding per head
        b, s, _ = x.size()
        # Reshape to (b, s, n_heads, head_dim)
        x_heads = x.view(b, s, self.n_heads, self.head_dim)
        # Apply RoPE
        x_heads = self.rope(x_heads)  # citeturn1search4
        # Restore shape
        x = x_heads.view(b, s, self.d_model)

        # 3) Transformer encoding
        x = self.transformer_encoder(x)

        # 4) Pool and classify
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)



class ClusterElasticGPC(ClassifierMixin,BaseEstimator):
    """
    Hybrid classifier: KMeans clustering → ElasticNet feature selection → GaussianProcessClassifier.
    """
    def __init__(
        self,
        n_clusters=5,
        en_alpha=1.0,
        en_l1_ratio=0.5,
        gp_kernel=None,
        random_state=42
    ):
        self.n_clusters = n_clusters
        self.en_alpha = en_alpha
        self.en_l1_ratio = en_l1_ratio
        self.gp_kernel = gp_kernel or 1.0 * RBF(length_scale=1.0)
        self.random_state = random_state

    def fit(self, X, y):
        # 1) Cluster the data
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        ).fit(X)  # :contentReference[oaicite:4]{index=4}
        clusters = self.kmeans_.labels_
        
        # Prepare per-cluster models
        self.elasticnets_ = {}
        self.gp_models_ = {}
        
        for c in range(self.n_clusters):
            idx = np.where(clusters == c)[0]
            Xc, yc = X[idx], y[idx]
            if len(idx) == 0:
                continue
            
            # 2) ElasticNet for feature selection
            en = ElasticNet(
                alpha=self.en_alpha,
                l1_ratio=self.en_l1_ratio,
                random_state=self.random_state
            )
            en.fit(Xc, yc)  # :contentReference[oaicite:5]{index=5}
            selected = np.where(en.coef_ != 0)[0]
            if selected.size == 0:
                # Fallback: use all features if none selected
                selected = np.arange(X.shape[1])
            self.elasticnets_[c] = selected
            
            # 3) Train GPC on selected features
            gp = GaussianProcessClassifier(
                kernel=self.gp_kernel,
                random_state=self.random_state
            )
            gp.fit(Xc[:, selected], yc)  # :contentReference[oaicite:6]{index=6}
            self.gp_models_[c] = gp
        
        return self

    def predict_proba(self, X):
        # Route each sample to its cluster’s GPC
        clusters = self.kmeans_.predict(X)
        proba = np.zeros((X.shape[0], 2))
        
        for c, gp in self.gp_models_.items():
            sel = np.where(clusters == c)[0]
            if sel.size == 0:
                continue
            idx_feats = self.elasticnets_[c]
            proba[sel] = gp.predict_proba(X[sel][:, idx_feats])
        
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

# Skorch wrapper
# net_rop = NeuralNetClassifier(
#     module=FraudTransformerWithRoPE,
#     module__input_dim=30,        # number of transaction features
#     module__d_model=64,
#     module__n_heads=4,
#     module__num_layers=2,
#     module__dropout=0.1,
#     module__num_classes=2,
#     module__max_seq_len=500,
#     module__rope_base=10000,
#     max_epochs=20,
#     lr=1e-3,
#     optimizer=torch.optim.Adam,
#     criterion=nn.CrossEntropyLoss,
#     batch_size=64,
#     device='cuda' if torch.cuda.is_available() else 'cpu',
# )

# Skorch wrapper for GRU
# gru_net = NeuralNetClassifier(
#     module=FraudGRU,
#     module__input_dim=30,    # Set to your number of transaction features
#     module__hidden_dim=64,
#     module__num_layers=2,
#     module__dropout=0.1,
#     module__num_classes=2,
#     max_epochs=20,
#     lr=1e-3,
#     optimizer=torch.optim.Adam,
#     criterion=nn.CrossEntropyLoss,
#     batch_size=64,
#     device='cuda' if torch.cuda.is_available() else 'cpu',
# )


# ---------- LSTM-based Model ----------


# # Skorch wrapper for LSTM
# lstm_net = NeuralNetClassifier(
#     module=FraudLSTM,
#     module__input_dim=30,    # Set to your number of transaction features
#     module__hidden_dim=64,
#     module__num_layers=2,
#     module__dropout=0.1,
#     module__num_classes=2,
#     max_epochs=20,
#     lr=1e-3,
#     optimizer=torch.optim.Adam,
#     criterion=nn.CrossEntropyLoss,
#     batch_size=64,
#     device='cuda' if torch.cuda.is_available() else 'cpu',
# )
# Example Skorch wrapper
# net = NeuralNetClassifier(
#     module=FraudTransformer,
#     module__input_dim=30,    # set to your num transaction features
#     module__d_model=64,
#     module__n_heads=4,
#     module__num_layers=2,
#     module__dropout=0.1,
#     module__num_classes=2,
#     max_epochs=20,
#     lr=1e-3,
#     optimizer=torch.optim.Adam,
#     criterion=nn.CrossEntropyLoss,
#     batch_size=64,
#     device='cuda' if torch.cuda.is_available() else 'cpu',
# )


