"""Encoder-decoder LSTM para pronosticar varias semanas a la vez."""
import torch
from torch import nn


class Seq2SeqLSTM(nn.Module):
    """El encoder resume las últimas `lookback` semanas en su estado oculto;
    el decoder parte de ese estado y genera una semana a la vez, usando su
    propia predicción anterior como entrada del siguiente paso."""

    def __init__(self, hidden_size=32, num_layers=1, dropout=0.2):
        super().__init__()
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(1, hidden_size, num_layers, batch_first=True, dropout=rnn_dropout)
        self.decoder = nn.LSTM(1, hidden_size, num_layers, batch_first=True, dropout=rnn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x, horizon, target=None, teacher_forcing=0.0):
        # x: (batch, lookback, 1)  ->  salida: (batch, horizon)
        _, state = self.encoder(x)
        step_input = x[:, -1:, :]  # el decoder arranca con la última semana observada
        outputs = []
        for step in range(horizon):
            out, state = self.decoder(step_input, state)
            pred = self.head(self.dropout(out[:, -1, :]))  # (batch, 1)
            outputs.append(pred)
            if target is not None and torch.rand(1).item() < teacher_forcing:
                step_input = target[:, step:step + 1].unsqueeze(-1)
            else:
                step_input = pred.unsqueeze(1)
        return torch.cat(outputs, dim=1)
