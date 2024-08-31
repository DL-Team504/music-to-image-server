from collections import Counter
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import math
import torchaudio
import torch.nn.functional as F


class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        return nltk.word_tokenize(text.lower())

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

                if frequencies[word] >= self.freq_threshold:
                    if word not in self.stoi:
                        self.stoi[word] = idx
                        self.itos[idx] = word
                        idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]


# 3. Dataset and DataLoader


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class AudioEncoder(nn.Module):
    def __init__(self, n_mels, d_model, nhead, num_encoder_layers, dim_feedforward):
        super(AudioEncoder, self).__init__()
        self.conv = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward),
            num_layers=num_encoder_layers,
        )
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, src):
        # src shape: (batch_size, time_steps, n_mels)
        src = self.conv(src.transpose(1, 2))  # (batch_size, d_model, time_steps)
        src = src.transpose(1, 2)  # (batch_size, time_steps, d_model)
        src = self.positional_encoding(src)  # (batch_size, time_steps, d_model)
        output = self.transformer_encoder(
            src.transpose(0, 1)
        )  # (time_steps, batch_size, d_model)
        return output.transpose(0, 1)  # (batch_size, time_steps, d_model)


class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward):
        super(CaptionDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward),
            num_layers=num_decoder_layers,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        # tgt shape: (batch_size, tgt_len)
        tgt = self.embedding(tgt)  # (batch_size, tgt_len, d_model)
        tgt = self.positional_encoding(
            tgt.transpose(0, 1)
        )  # (tgt_len, batch_size, d_model)
        output = self.transformer_decoder(
            tgt, memory.transpose(0, 1)
        )  # (tgt_len, batch_size, d_model)
        output = self.fc_out(output)  # (tgt_len, batch_size, vocab_size)
        return output.transpose(0, 1)  # (batch_size, tgt_len, vocab_size)


class AudioCaptioningModel(nn.Module):
    def __init__(
        self,
        n_mels,
        vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
    ):
        super(AudioCaptioningModel, self).__init__()
        self.encoder = AudioEncoder(
            n_mels, d_model, nhead, num_encoder_layers, dim_feedforward
        )
        self.decoder = CaptionDecoder(
            vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward
        )
        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        memory = self.encoder(src)

        # Shift the target to the left (remove the last token for input)
        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]  # The expected output, shifted by one

        # Embed and apply positional encoding to the target sequence
        tgt_embedded = self.decoder.embedding(tgt_input)
        tgt_embedded = self.decoder.positional_encoding(
            tgt_embedded.transpose(0, 1)
        )  # (tgt_len, batch_size, d_model)

        # Generate a causal mask for the decoder (prevent attending to future tokens)
        tgt_len = tgt_input.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(
            tgt.device
        )  # (tgt_len, tgt_len)

        # Generate padding mask (optional if padding is used)
        if (
            hasattr(self.decoder.embedding, "padding_idx")
            and self.decoder.embedding.padding_idx is not None
        ):
            # Create the padding mask as a boolean tensor
            tgt_padding_mask = (
                tgt_input == self.decoder.embedding.padding_idx
            )  # (batch_size, tgt_len)
        else:
            tgt_padding_mask = None

        # Apply the transformer decoder
        output = self.decoder.transformer_decoder(
            tgt_embedded,
            memory.transpose(0, 1),
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        # Apply the final linear layer
        output = self.decoder.fc_out(output)  # (tgt_len, batch_size, vocab_size)
        return output.transpose(0, 1)  # (batch_size, tgt_len, vocab_size)


def infer(
    audio_path,
    # waveform,
    # sample_rate,
    model,
    vocab,
    fixed_length=160000,
    n_mels=128,
    target_sample_rate=16000,
    max_caption_length=50,
    device="cuda",
):
    # Load and preprocess the audio
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
    waveform = torchaudio.transforms.Resample(
        orig_freq=sample_rate, new_freq=target_sample_rate
    )(waveform)

    # Pad or truncate waveform to fixed length
    if waveform.size(1) > fixed_length:
        waveform = waveform[:, :fixed_length]
    else:
        pad_length = fixed_length - waveform.size(1)
        waveform = F.pad(waveform, (0, pad_length))

    # Convert waveform to mel spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate, n_mels=n_mels
    )(waveform)

    # Prepare the input tensor for the model
    mel_spectrogram = (
        mel_spectrogram.squeeze(0).T.unsqueeze(0).to(device)
    )  # (1, time_steps, n_mels)

    # Set the model to evaluation mode
    model.eval()

    # Initialize the input for the decoder (start with the <SOS> token)
    tgt_input = torch.tensor([[vocab.stoi["<SOS>"]]], device=device)  # Shape: (1, 1)

    # Run the encoder
    with torch.no_grad():
        memory = model.encoder(mel_spectrogram)

    # Initialize a list to store generated tokens
    generated_tokens = []

    # Generate the caption using the decoder
    for _ in range(max_caption_length):
        # Embed and apply positional encoding to the target input
        tgt_embedded = model.decoder.embedding(tgt_input)
        tgt_embedded = model.decoder.positional_encoding(
            tgt_embedded.transpose(0, 1)
        )  # (tgt_len, batch_size, d_model)

        # Generate a causal mask for the decoder
        tgt_len = tgt_input.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(
            device
        )  # (tgt_len, tgt_len)

        # Run the decoder
        output = model.decoder.transformer_decoder(
            tgt_embedded,
            memory.transpose(0, 1),
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=None,  # No padding mask needed during inference
        )

        # Apply the final linear layer to get logits
        output = model.decoder.fc_out(output)  # (tgt_len, batch_size, vocab_size)

        # Get the predicted token for the current step
        next_token = output[-1, :, :].argmax(-1).unsqueeze(0)  # Shape: (1, batch_size)

        # Transpose next_token to match tgt_input shape for concatenation
        next_token = next_token.transpose(0, 1)  # Shape: (batch_size, 1)

        # Append the predicted token to the input sequence and to the generated tokens
        tgt_input = torch.cat([tgt_input, next_token], dim=1)
        generated_tokens.append(next_token.item())

        # Stop if <EOS> token is generated
        if next_token.item() == vocab.stoi["<EOS>"]:
            break

    # Convert the generated sequence of tokens to words
    generated_caption = [vocab.itos[token] for token in generated_tokens]

    return " ".join(generated_caption)
