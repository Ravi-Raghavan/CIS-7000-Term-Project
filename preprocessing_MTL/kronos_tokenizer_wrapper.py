import sys
import os
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Kronos"))
from model.kronos import KronosTokenizer

KRONOS_EMBED_DIM = 256
KRONOS_OUTPUT_DIM = 320
VOCAB_SIZE = 1024

class KronosEncoderWrapper(nn.Module):
    def __init__(self, device, kronos_model_name="NeoQuasar/Kronos-Tokenizer-base",
                 lora_path: str = None):
        super().__init__()
        self.device = device
        self.tokenizer = KronosTokenizer.from_pretrained(kronos_model_name)
        if lora_path:
            from peft import PeftModel
            self.tokenizer = PeftModel.from_pretrained(self.tokenizer, lora_path)
            print(f"Loaded LoRA adapter from: {lora_path}")
        self.tokenizer.eval()
        self.tokenizer.to(device)
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)

        self.emb_s1 = nn.Embedding(VOCAB_SIZE, KRONOS_EMBED_DIM)
        self.emb_s2 = nn.Embedding(VOCAB_SIZE, KRONOS_EMBED_DIM)
        # s1-only projection → shared (market-level) path
        self.proj_shared  = nn.Linear(KRONOS_EMBED_DIM, KRONOS_OUTPUT_DIM)
        # s1+s2 projection → private (stock-specific) path
        self.proj_private = nn.Linear(KRONOS_EMBED_DIM * 2, KRONOS_OUTPUT_DIM)

    def precompute_tokens(self, raw_ohlcva: np.ndarray, chunk_size=512):
        """
        Runs frozen tokenizer over full time series.
        Args:
            raw_ohlcva: (K, T, 6) float32 — log1p-scaled, clipped ±5
            chunk_size: max timesteps per forward pass (≤512 for Tokenizer-base)
        Returns:
            s1_tokens: (K, T) int64 numpy
            s2_tokens: (K, T) int64 numpy
        """
        K, T, C = raw_ohlcva.shape
        s1_all = []
        s2_all = []
        
        self.tokenizer.eval()
        with torch.no_grad():
            for k in range(K):
                stock_data = torch.tensor(raw_ohlcva[k], dtype=torch.float32).to(self.device) # (T, 6)
                stock_s1 = []
                stock_s2 = []
                for i in range(0, T, chunk_size):
                    chunk = stock_data[i:i+chunk_size].unsqueeze(0) # (1, chunk_size, 6)
                    s1, s2 = self.tokenizer.encode(chunk, half=True)
                    stock_s1.append(s1.cpu().numpy()[0])
                    stock_s2.append(s2.cpu().numpy()[0])
                s1_all.append(np.concatenate(stock_s1, axis=0)) # (T,)
                s2_all.append(np.concatenate(stock_s2, axis=0)) # (T,)
                
        return np.array(s1_all, dtype=np.int64), np.array(s2_all, dtype=np.int64)

    def forward(self, s1: torch.Tensor, s2: torch.Tensor):
        """
        Hierarchical token → embedding projection.
        Args:
            s1: (B, K, T) long  — coarse BSQ tokens
            s2: (B, K, T) long  — fine BSQ tokens (refinement on top of s1)
        Returns:
            x_shared:  (B, K, T, 320)  — s1 only, feeds SharedDecoder
            x_private: (B, K, T, 320)  — s1+s2,  feeds PrivateDecoder
        """
        B, K, T = s1.shape
        e1 = self.emb_s1(s1.view(B * K, T))                        # (B*K, T, 256)
        e2 = self.emb_s2(s2.view(B * K, T))                        # (B*K, T, 256)
        x_shared  = self.proj_shared(e1)                            # (B*K, T, 320)
        x_private = self.proj_private(torch.cat([e1, e2], dim=-1))  # (B*K, T, 320)
        return x_shared.view(B, K, T, 320), x_private.view(B, K, T, 320)
