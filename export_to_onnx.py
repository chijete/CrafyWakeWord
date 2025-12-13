# export_to_onnx.py
# Versión Final: 
# 1. ConvSTFT para evitar errores de STFT/Complex.
# 2. Reemplazo de AdaptiveAvgPool2d por torch.mean para compatibilidad ONNX.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import json
import os
import numpy as np
import math

# -----------------------------------------------------------
# 1. ARQUITECTURA DEL MODELO (Adaptada para ONNX)
# -----------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Sequential()
        should_downsample = (stride != 1) if isinstance(stride, int) else (stride != (1,1))
        if should_downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        w = self.attention(x)
        return torch.sum(w * x, dim=1)

class SpeechResModel(nn.Module):
    def __init__(self, num_labels, n_maps=48):
        super().__init__()
        self.conv0 = nn.Conv2d(1, n_maps, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.res1 = ResBlock(n_maps, n_maps, stride=1)
        self.res2 = ResBlock(n_maps, n_maps, stride=(2, 2))
        self.res3 = ResBlock(n_maps, n_maps, stride=1)
        self.res4 = ResBlock(n_maps, n_maps, stride=(2, 2))
        
        # ELIMINADO: self.pool_freq = nn.AdaptiveAvgPool2d((1, None))
        # Reemplazaremos esto con torch.mean en el forward para compatibilidad ONNX
        
        self.attn = AttentionPooling(n_maps)
        self.fc = nn.Sequential(
            nn.Linear(n_maps, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_labels)
        )

    def forward(self, x):
        # x: (Batch, 1, Freq, Time)
        x = self.conv0(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x) 
        # Output shape aquí: (Batch, n_maps, FreqReducida, TimeReducido)
        
        # CORRECCIÓN PARA ONNX:
        # En lugar de AdaptiveAvgPool2d((1, None)), hacemos la media sobre el eje de Frecuencia (dim 2).
        # Esto colapsa la frecuencia a 1, manteniendo el tiempo.
        # (B, C, F, T) -> (B, C, 1, T)
        x = torch.mean(x, dim=2, keepdim=True)
        
        x = x.squeeze(2)       # (B, C, T)
        x = x.permute(0, 2, 1) # (B, T, C)
        x = self.attn(x)       # (B, C)
        return self.fc(x)

# -----------------------------------------------------------
# 2. STFT BASADO EN CONVOLUCIÓN (ONNX SAFE)
# -----------------------------------------------------------
class ConvSTFT(nn.Module):
    def __init__(self, n_fft, hop_length, win_length, window):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float32)
        n = torch.arange(win_length, dtype=torch.float32)
        angle = 2 * math.pi * k.unsqueeze(1) * n.unsqueeze(0) / n_fft
        dft_real = torch.cos(angle)
        dft_imag = -torch.sin(angle)
        
        if window is not None:
            w = window.unsqueeze(0)
            dft_real = dft_real * w
            dft_imag = dft_imag * w
            
        self.out_channels_complex = n_fft // 2 + 1
        kernels = torch.cat([dft_real, dft_imag], dim=0) 
        kernels = kernels.unsqueeze(1)
        self.register_buffer('weight', kernels)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        stft_out = F.conv1d(x, self.weight, stride=self.hop_length)
        real_part = stft_out[:, :self.out_channels_complex, :]
        imag_part = stft_out[:, self.out_channels_complex:, :]
        return real_part.pow(2) + imag_part.pow(2)

# -----------------------------------------------------------
# 3. WRAPPER "END-TO-END"
# -----------------------------------------------------------
class InferenceWrapper(nn.Module):
    def __init__(self, model, sr, n_fft, n_mels, mean, std):
        super().__init__()
        self.model = model
        
        ref_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, n_mels=n_mels, center=False
        )
        
        self.stft = ConvSTFT(
            n_fft=n_fft, 
            hop_length=ref_mel.hop_length, 
            win_length=ref_mel.win_length, 
            window=ref_mel.spectrogram.window
        )
        
        self.register_buffer('mel_basis', ref_mel.mel_scale.fb)
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))

    def forward(self, audio):
        power_spec = self.stft(audio)
        power_spec_t = power_spec.transpose(1, 2)
        mel_spec = torch.matmul(power_spec_t, self.mel_basis)
        mel_spec = mel_spec.transpose(1, 2)
        
        log_mel = (mel_spec + 1e-6).log()
        normed = (log_mel - self.mean) / (self.std + 1e-9)
        normed = normed.unsqueeze(1)
        
        logits = self.model(normed)
        return torch.softmax(logits, dim=1)

# -----------------------------------------------------------
# 4. EXPORTACIÓN
# -----------------------------------------------------------
def main():
    CHECKPOINT_DIR = "dataset/checkpoints"
    MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model_best.pt")
    JSON_PATH = os.path.join(CHECKPOINT_DIR, "model_best.json")
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(JSON_PATH):
        print("Error: No se encuentran los archivos model_best.pt o .json")
        return

    with open(JSON_PATH, 'r') as f:
        meta = json.load(f)
        
    sr = meta['audio_config']['sample_rate']
    window_ms = meta['audio_config']['window_ms']
    input_len = int(sr * (window_ms / 1000))
    classes = meta['labels']['classes']
    
    print(f"Configuración: SR={sr}, Window={window_ms}ms, Clases={classes}")

    device = torch.device('cpu')
    base_model = SpeechResModel(num_labels=len(classes))
    # Cargar pesos ignorando la diferencia en pool_freq (que ahora no existe como capa)
    # strict=False es necesario porque hemos quitado la capa pool_freq, 
    # pero como esa capa no tenía pesos entrenables (era solo una función), no pasa nada.
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    base_model.eval()

    full_model = InferenceWrapper(
        base_model, 
        sr=sr,
        n_fft=meta['audio_config']['n_fft'],
        n_mels=meta['audio_config']['n_mels'],
        mean=meta['normalization']['zmuv_mean'],
        std=meta['normalization']['zmuv_std']
    ).to(device)

    output_dir = "webapp/model"
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "wakeword_model.onnx")
    
    dummy_input = torch.randn(1, input_len)
    
    print(f"Exportando a ONNX (Opset 12)...")
    
    torch.onnx.export(
        full_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12, 
        do_constant_folding=True,
        input_names=['audio_input'],
        output_names=['probabilities'],
        dynamic_axes={'audio_input': {0: 'batch_size'}, 'probabilities': {0: 'batch_size'}}
    )
    print("✅ Modelo exportado correctamente.")

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quant_path = os.path.join(output_dir, "wakeword_model.quant.onnx")
        print(f"Cuantizando a Int8...")
        quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QUInt8)
        print("✅ Modelo Cuantizado listo.")
    except ImportError:
        print("AVISO: Instala onnxruntime para cuantizar.")

if __name__ == "__main__":
    main()