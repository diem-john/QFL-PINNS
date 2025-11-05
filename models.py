# models.py

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.init as init


class Seq2Seq(nn.Module):
    """
    Typhoon Track Forecasting Model (Encoder-Decoder with Attention)
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, num_heads: int = 4):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    init.orthogonal_(param)
                elif 'linear' in name:
                    init.xavier_uniform_(param)
            elif 'bias' in name:
                init.constant_(param, 0)

    def forward(self, x: Tensor, horizon_length: int) -> Tensor:
        encoder_output, (hidden, cell) = self.encoder(x)
        decoder_input = torch.zeros(x.size(0), horizon_length, self.hidden_size, device=x.device)
        decoder_out, _ = self.decoder(decoder_input, (hidden, cell))
        attention_output, _ = self.multihead_attention(
            query=decoder_out,
            key=encoder_output,
            value=encoder_output
        )
        final_output = self.fc(attention_output)
        return final_output


class HybridModelPINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for Wind Speed Forecasting
    """

    def __init__(self, input_size: int, output_size: int, sequence_length: int,
                 embedding_dim: int = 48, n_head: int = 4, num_transformer_layers: int = 4,
                 conv_channels: int = 64, kernel_size: int = 3, pool_kernel: int = 2,
                 hidden_dense: int = 512):
        super(HybridModelPINN, self).__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        self.linear_projection = nn.Linear(input_size, embedding_dim)
        self.positional_encoding = self._get_positional_encoding(sequence_length, embedding_dim)
        self.register_buffer('positional_encoding_buffer', self.positional_encoding)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_head, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, num_layers=num_transformer_layers
        )

        self.conv_layer = nn.Conv1d(embedding_dim, conv_channels, kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.pooling_layer = nn.MaxPool1d(pool_kernel)

        conv_output_length = sequence_length
        pooled_output_length = conv_output_length // pool_kernel
        flattened_size = conv_channels * pooled_output_length + embedding_dim * sequence_length

        self.dense1 = nn.Linear(flattened_size, hidden_dense)
        self.dense2 = nn.Linear(hidden_dense, output_size)

        self.u_dense = nn.Linear(hidden_dense, output_size)
        self.v_dense = nn.Linear(hidden_dense, output_size)
        self.p_dense = nn.Linear(hidden_dense, output_size)

        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)
            elif 'bias' in name:
                init.constant_(param, 0)

    def _get_positional_encoding(self, seq_len: int, d_model: int) -> Tensor:
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe.squeeze(1).unsqueeze(0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        embedded = self.linear_projection(x)
        positional_encoding = self.positional_encoding_buffer[:, :embedded.size(1), :].to(embedded.device)
        embedded = embedded + positional_encoding

        transformer_out = self.transformer_encoder(embedded)
        transformer_out_flattened = transformer_out.reshape(transformer_out.size(0), -1)

        cnn_in = embedded.permute(0, 2, 1)
        conv_out = self.relu(self.conv_layer(cnn_in))
        pooled_out = self.pooling_layer(conv_out)
        pooled_out_flattened = pooled_out.reshape(pooled_out.size(0), -1)

        combined_features = torch.cat((pooled_out_flattened, transformer_out_flattened), dim=1)
        dense1_out = self.relu(self.dense1(combined_features))

        output = self.dense2(dense1_out)
        u_out = self.u_dense(dense1_out)
        v_out = self.v_dense(dense1_out)
        p_out = self.p_dense(dense1_out)

        return output, u_out, v_out, p_out