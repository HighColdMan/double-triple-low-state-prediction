import torch
import torch.nn as nn
import math

class CNN_LSTM_GRU_ResNet_Model(nn.Module):
    def __init__(self, input_size, hidden_size_lstm, hidden_size_gru, output_size):
        super(CNN_LSTM_GRU_ResNet_Model, self).__init__()

        # 1D-CNN 模块 (多尺度卷积)
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_size, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_size, 32, kernel_size=7, padding=3)
        self.bn_cnn = nn.BatchNorm1d(96)  # 多尺度输出拼接后的通道数

        # LSTM 和 GRU 层
        self.lstm = nn.LSTM(96, hidden_size_lstm, batch_first=True, num_layers=2, dropout=0.3)
        self.gru = nn.GRU(hidden_size_lstm, hidden_size_gru, batch_first=True, num_layers=2, dropout=0.3)
        
        # 全连接层输出
        self.fc = nn.Sequential(
            nn.Linear(hidden_size_gru, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        # 输入 x: (batch_size, time_steps, features)
        x = x.permute(0, 2, 1)  # 调整为 (batch_size, features, time_steps)
        # 多尺度卷积
        x1 = torch.relu(self.conv1(x))
        x2 = torch.relu(self.conv2(x))
        x3 = torch.relu(self.conv3(x))
        x = torch.cat([x1, x2, x3], dim=1)  # 拼接多尺度特征
        
        x = self.bn_cnn(x)  # BatchNorm
        x = x.permute(0, 2, 1)  # 调整回 (batch_size, time_steps, features)
        
        # LSTM 层
        x, _ = self.lstm(x)
        
        # GRU 层
        x, _ = self.gru(x)
        
        # 取最后时间步的输出
        x = x[:, -1, :]  # (batch_size, hidden_size_gru)
        x = self.fc(x)   # 全连接层输出

        return x
    

# ==========================
# Positional Encoding
# ==========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len].to(x.device)


# ==========================
# LayerNorm CNN Block
# ==========================
class LNConvBlock(nn.Module):
    """
    Conv1d → GELU → LayerNorm → Conv1d → GELU → LayerNorm
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=kernel_size, padding=padding)
        self.act1 = nn.GELU()
        self.ln1 = nn.LayerNorm(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=padding)
        self.act2 = nn.GELU()
        self.ln2 = nn.LayerNorm(out_channels)

    def forward(self, x):
        """
        x: (B, C, T) → LN 需要 (B, T, C)
        """
        x = self.conv1(x)
        x = self.act1(x)
        x = x.permute(0, 2, 1)  # (B,T,C)
        x = self.ln1(x)
        x = x.permute(0, 2, 1)

        x = self.conv2(x)
        x = self.act2(x)
        x = x.permute(0, 2, 1)
        x = self.ln2(x)
        x = x.permute(0, 2, 1)

        return x


# ==========================
# Optimized MS_CNN_Transformer
# ==========================
class MS_CNN_Transformer(nn.Module):

    def __init__(self, input_dim, cnn_channels=128, d_model=256,
                 nhead=8, num_layers=4,
                 num_outputs_reg=3, num_outputs_cls=5, dropout=0.2):

        super().__init__()

        # 三尺度 CNN（全部 LN + GELU）
        self.conv3 = LNConvBlock(input_dim, cnn_channels, 3, padding=1)
        self.conv5 = LNConvBlock(input_dim, cnn_channels, 5, padding=2)
        self.conv7 = LNConvBlock(input_dim, cnn_channels, 7, padding=3)

        # CNN 合并后 LayerNorm
        self.pre_transformer_ln = nn.LayerNorm(cnn_channels * 3)

        # 映射到 Transformer 维度
        self.proj = nn.Linear(cnn_channels * 3, d_model)

        # Transformer
        self.pos_encoder = PositionalEncoding(d_model)
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=512,
            nhead=nhead,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)

        # Regression Heads
        self.mbp_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.other_reg_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_outputs_reg - 1)
        )

        # Classification Head
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_outputs_cls)
        )

    def forward(self, x):
        """
        x: (B, seq, features)
        """
        x = x.permute(0, 2, 1)  # (B, F, T)

        # Multi-Scale CNN
        x1 = self.conv3(x)
        x2 = self.conv5(x)
        x3 = self.conv7(x)

        x = torch.cat([x1, x2, x3], dim=1)  # (B, 3C, T)

        # Transformer 期望 (B,T,Features)
        x = x.permute(0, 2, 1)
        x = self.pre_transformer_ln(x)

        x = self.proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        # mean pooling (优于 last)
        feat = x.mean(dim=1)

        # Multi-task output
        mbp = self.mbp_head(feat)
        other = self.other_reg_head(feat)
        reg_out = torch.cat([mbp, other], dim=1)

        cls_out = self.cls_head(feat)

        return reg_out, cls_out
