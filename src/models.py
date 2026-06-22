"""Definizione dei modelli e factory `build_model`.

Implementazione CNN custom (baseline didattica). I modelli
pretrained (ResNet/EfficientNet/Swin) verranno aggiunti qui in M3/M4 ampliando
`build_model`, senza toccare il loop di training (che resta generico sul modello).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Blocco convoluzionale base: Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2.

    - in_ch: canali in ingresso (es. 3 per immagini RGB: 3 x 224 x 224)
    - out_ch: canali in uscita
    - Conv 3x3 con padding=1: estrae feature locali mantenendo la dimensione spaziale (32 x 224 x 224).
    - BatchNorm: normalizza le attivazioni -> training piu' stabile e veloce.
    - ReLU: non linearita'.
    - MaxPool 2x2 (prende matrici 2x2 e seleazioni il valore localmente migliore): dimezza altezza e larghezza (sotto-campionamento: 32 x 112 x 112).
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class CustomCNN(nn.Module):
    """CNN convoluzionale (baseline real/fake).

    Architettura (input 3x224x224):
        blocco 1: 3   -> 32   | 224 -> 112
        blocco 2: 32  -> 64   | 112 -> 56
        blocco 3: 64  -> 128  | 56  -> 28
        blocco 4: 128 -> 256  | 28  -> 14
        
        AdaptiveAvgPool -> 256x1x1 -> flatten -> Dropout -> Linear(256, num_classes), per ogni canale finale 
        256 x 14 x 14 si prende il valore medio (AdaptiveAvgPool) -> 256 -> classificazione.

    L'AdaptiveAvgPool rende la testa indipendente dalla risoluzione esatta in ingresso
    (riduce ogni feature map a un singolo valore medio): robusto e con pochi parametri.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3,
                 channels: tuple[int, ...] = (32, 64, 128, 256)) -> None:
        super().__init__() #necessario utilizzare correttamente nn.Module

        # --- Estrattore di feature: sequenza di blocchi convoluzionali ---
        blocks = []
        in_ch = 3  # immagine RGB
        for out_ch in channels:
            blocks.append(conv_block(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)      # *blocks: unpack della lista in argomenti posizionali

        # --- Testa di classificazione ---
        self.pool = nn.AdaptiveAvgPool2d(1)          # -> (256, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                            # -> (256)
            nn.Dropout(dropout),                     # regolarizzazione
            nn.Linear(in_ch, num_classes),           # -> (B, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Factory: costruisce il modello a partire dalla sezione `model` del config.

    Il campo `model.name` seleziona l'architettura. Aggiungere qui i nuovi modelli
    (resnet50, efficientnet_b0, swin_tiny) in M3/M4.
    """
    model_cfg = cfg["model"]
    name = model_cfg["name"]
    num_classes = model_cfg.get("num_classes", 2)

    if name == "cnn_custom":
        return CustomCNN(
            num_classes=num_classes,
            dropout=model_cfg.get("dropout", 0.3),
        )

    raise NotImplementedError(
        f"Modello '{name}' non ancora implementato "
        f"Disponibili al momento: ['cnn_custom']."
    )


if __name__ == "__main__":
    # Ispezione rapida: architettura, n. parametri e shape di output su input fittizio.
    cfg = {"model": {"name": "cnn_custom", "num_classes": 2, "dropout": 0.3}}
    model = build_model(cfg)
    dummy = torch.randn(2, 3, 224, 224)  # batch di 2 immagini RGB 224x224
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"\nParametri allenabili: {n_params:,}")
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}  (logit per [nature, ai])")
