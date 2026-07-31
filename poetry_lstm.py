import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------
# 1. CHARACTER-LEVEL LSTM
# --------------------------------------------------------
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

# --------------------------------------------------------
# 2. WORD-LEVEL LSTM WITH ATTENTION
# --------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs):
        attn_weights = F.softmax(self.attn(lstm_outputs), dim=1)
        context = torch.sum(attn_weights * lstm_outputs, dim=1)
        return context, attn_weights

class WordLSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super(WordLSTMWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        output = self.fc(context)
        return output, attn_weights

# --------------------------------------------------------
# 3. SCALED DOT-PRODUCT ATTENTION (Transformer/Vision Core)
# --------------------------------------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model=256):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights

# --------------------------------------------------------
# SANITY CHECK (Runs to prove models work)
# --------------------------------------------------------
if __name__ == "__main__":
    print("--- Initializing Models for Portfolio ---")
    
    char_model = CharLSTM(vocab_size=50)
    dummy_input = torch.randint(0, 50, (2, 10))
    output = char_model(dummy_input)
    print(f"Char LSTM output shape: {output.shape}")

    word_model = WordLSTMWithAttention(vocab_size=5000)
    dummy_words = torch.randint(0, 5000, (2, 15))
    out, attn = word_model(dummy_words)
    print(f"Word LSTM+Attn output shape: {out.shape}")

    sa = ScaledDotProductAttention()
    d = torch.randn(2, 10, 256)
    attn_out, weights = sa(d, d, d)
    print(f"Transformer Self-Attention output shape: {attn_out.shape}")
    
    print("\n--- Models ready! ---")
