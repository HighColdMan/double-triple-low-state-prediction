import torch
import torch.nn as nn


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