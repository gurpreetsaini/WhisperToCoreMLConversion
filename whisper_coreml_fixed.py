#!/usr/bin/env python3
"""
Whisper to Core ML Converter with ANE Optimization (SSL Fixed Version)

Converts OpenAI Whisper models to Apple Core ML format with optional
Apple Neural Engine (ANE) optimizations for better performance on Apple devices.

Requirements:
- torch
- coremltools
- openai-whisper
- ane_transformers (optional, for ANE optimization)

Usage:
    python whisper_coreml_fixed.py --model tiny --quantize --optimize-ane
    python whisper_coreml_fixed.py --model base.en --encoder-only
"""

import argparse
import os
import sys
import ssl
import urllib.request
from pathlib import Path
from typing import Dict, Optional

# SSL fix for macOS
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

original_urlopen = urllib.request.urlopen
def patched_urlopen(*args, **kwargs):
    if 'context' not in kwargs:
        kwargs['context'] = ssl_context
    return original_urlopen(*args, **kwargs)

urllib.request.urlopen = patched_urlopen

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    import coremltools as ct
    from coremltools.models.neural_network.quantization_utils import quantize_weights
except ImportError:
    print("Error: coremltools not installed. Install with: pip install coremltools")
    sys.exit(1)

try:
    import whisper
    from whisper import load_model
    from whisper.model import (
        AudioEncoder, 
        ModelDimensions, 
        MultiHeadAttention, 
        ResidualAttentionBlock, 
        TextDecoder, 
        Whisper
    )
except ImportError:
    print("Error: openai-whisper not installed. Install with: pip install openai-whisper")
    sys.exit(1)

# Try to import ANE transformers (optional)
try:
    from ane_transformers.reference.layer_norm import LayerNormANE as LayerNormANEBase
    ANE_AVAILABLE = True
except ImportError:
    print("Warning: ane_transformers not available. ANE optimizations disabled.")
    print("Install with: pip install ane_transformers")
    ANE_AVAILABLE = False
    LayerNormANEBase = nn.LayerNorm

# Disable PyTorch Scaled Dot-Product Attention for compatibility
import whisper.model
whisper.model.MultiHeadAttention.use_sdpa = False


def linear_to_conv2d_map(state_dict, prefix, local_metadata, strict,
                         missing_keys, unexpected_keys, error_msgs):
    """Convert nn.Linear weights to nn.Conv2d weights by adding dimensions."""
    for k in state_dict:
        is_attention = all(substr in k for substr in ['attn', '.weight'])
        is_mlp = any(k.endswith(s) for s in ['mlp.0.weight', 'mlp.2.weight'])
        
        if (is_attention or is_mlp) and len(state_dict[k].shape) == 2:
            state_dict[k] = state_dict[k][:, :, None, None]


def correct_for_bias_scale_order_inversion(state_dict, prefix, local_metadata,
                                           strict, missing_keys, unexpected_keys, error_msgs):
    """Correct bias scaling for ANE layer norm."""
    if prefix + 'bias' in state_dict and prefix + 'weight' in state_dict:
        state_dict[prefix + 'bias'] = state_dict[prefix + 'bias'] / state_dict[prefix + 'weight']
    return state_dict


class LayerNormANE(LayerNormANEBase):
    """ANE-optimized LayerNorm with bias correction."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if ANE_AVAILABLE:
            self._register_load_state_dict_pre_hook(correct_for_bias_scale_order_inversion)


class MultiHeadAttentionANE(MultiHeadAttention):
    """ANE-optimized MultiHeadAttention using Conv2d layers."""
    
    def __init__(self, n_state: int, n_head: int):
        super().__init__(n_state, n_head)
        # Replace Linear layers with Conv2d for ANE optimization
        self.query = nn.Conv2d(n_state, n_state, kernel_size=1)
        self.key = nn.Conv2d(n_state, n_state, kernel_size=1, bias=False)
        self.value = nn.Conv2d(n_state, n_state, kernel_size=1)
        self.out = nn.Conv2d(n_state, n_state, kernel_size=1)

    def forward(self, x: Tensor, xa: Optional[Tensor] = None, 
                mask: Optional[Tensor] = None, kv_cache: Optional[dict] = None):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention_ane(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention_ane(self, q: Tensor, k: Tensor, v: Tensor, 
                          mask: Optional[Tensor] = None):
        """ANE-optimized attention computation."""
        _, dim, _, seqlen = q.size()
        dim_per_head = dim // self.n_head
        scale = float(dim_per_head) ** -0.5
        q = q * scale

        # Split into multiple heads
        mh_q = q.split(dim_per_head, dim=1)
        mh_k = k.transpose(1, 3).split(dim_per_head, dim=3)
        mh_v = v.split(dim_per_head, dim=1)

        # Compute attention for each head
        mh_qk = [
            torch.einsum('bchq,bkhc->bkhq', [qi, ki])
            for qi, ki in zip(mh_q, mh_k)
        ]

        # Apply mask if provided
        if mask is not None:
            for head_idx in range(self.n_head):
                mh_qk[head_idx] = mh_qk[head_idx] + mask[:, :seqlen, :, :seqlen]

        # Apply softmax and compute weighted values
        attn_weights = [aw.softmax(dim=1) for aw in mh_qk]
        attn = [
            torch.einsum('bkhq,bchk->bchq', wi, vi) 
            for wi, vi in zip(attn_weights, mh_v)
        ]
        attn = torch.cat(attn, dim=1)

        return attn, torch.cat(mh_qk, dim=1).float().detach()


class ResidualAttentionBlockANE(ResidualAttentionBlock):
    """ANE-optimized ResidualAttentionBlock."""
    
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__(n_state, n_head, cross_attention)
        # Replace with ANE-optimized components
        self.attn = MultiHeadAttentionANE(n_state, n_head)
        self.attn_ln = LayerNormANE(n_state)
        
        if cross_attention:
            self.cross_attn = MultiHeadAttentionANE(n_state, n_head)
            self.cross_attn_ln = LayerNormANE(n_state)

        # ANE-optimized MLP using Conv2d
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Conv2d(n_state, n_mlp, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(n_mlp, n_state, kernel_size=1)
        )
        self.mlp_ln = LayerNormANE(n_state)


class AudioEncoderANE(AudioEncoder):
    """ANE-optimized AudioEncoder."""
    
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__(n_mels, n_ctx, n_state, n_head, n_layer)
        self.blocks = nn.ModuleList([
            ResidualAttentionBlockANE(n_state, n_head) for _ in range(n_layer)
        ])
        self.ln_post = LayerNormANE(n_state)

    def forward(self, x: Tensor):
        """Forward pass with ANE-optimized tensor shapes."""
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        # Add positional embedding and dummy dimension for ANE
        x = (x + self.positional_embedding.transpose(0, 1)).to(x.dtype).unsqueeze(2)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x.squeeze(2).transpose(1, 2)


class TextDecoderANE(TextDecoder):
    """ANE-optimized TextDecoder with chunked token processing."""
    
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__(n_vocab, n_ctx, n_state, n_head, n_layer)
        self.blocks = nn.ModuleList([
            ResidualAttentionBlockANE(n_state, n_head, cross_attention=True) 
            for _ in range(n_layer)
        ])
        self.ln = LayerNormANE(n_state)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """Forward pass with chunked token processing for large vocabularies."""
        offset = next(iter(kv_cache.values())).shape[3] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset:offset + x.shape[-1]]
        x = x.to(xa.dtype)

        # Reformat for ANE processing
        mask = self.mask[None, None, :, :].permute(0, 3, 1, 2)
        x = x.transpose(1, 2).unsqueeze(2)

        for block in self.blocks:
            x = block(x, xa, mask=mask, kv_cache=kv_cache)

        x = self.ln(x)
        x = x.permute(0, 2, 3, 1).squeeze(0)

        # Handle large vocabularies by chunking (ANE limitation: max 16,384)
        vocab_size = self.token_embedding.weight.shape[0]
        if vocab_size >= 51865:
            # Multilingual model - split into 11 chunks
            chunk_size = vocab_size // 11
            splits = self.token_embedding.weight.split(chunk_size, dim=0)
        elif vocab_size == 51864:
            # English model - split into 12 chunks
            chunk_size = vocab_size // 12
            splits = self.token_embedding.weight.split(chunk_size, dim=0)
        else:
            # Small vocabulary - no chunking needed
            return torch.einsum('bid,jd->bij', x, self.token_embedding.weight)

        # Compute logits in chunks and concatenate
        logits = torch.cat([
            torch.einsum('bid,jd->bij', x, split) for split in splits
        ]).view(*x.shape[:2], -1)

        return logits


class WhisperANE(Whisper):
    """ANE-optimized Whisper model."""
    
    def __init__(self, dims: ModelDimensions):
        super().__init__(dims)
        self.encoder = AudioEncoderANE(
            dims.n_mels, dims.n_audio_ctx, dims.n_audio_state,
            dims.n_audio_head, dims.n_audio_layer
        )
        self.decoder = TextDecoderANE(
            dims.n_vocab, dims.n_text_ctx, dims.n_text_state,
            dims.n_text_head, dims.n_text_layer
        )
        self._register_load_state_dict_pre_hook(linear_to_conv2d_map)

    def forward(self, mel: Tensor, tokens: Tensor) -> Dict[str, Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """Install hooks for key-value caching during inference."""
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if (module not in cache or 
                output.shape[3] > self.decoder.positional_embedding.shape[0]):
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=3).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttentionANE):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks


def convert_encoder(hparams: ModelDimensions, model: nn.Module, quantize: bool = False):
    """Convert audio encoder to Core ML format."""
    print("Converting encoder...")
    model.eval()

    # Create sample input
    input_shape = (1, hparams.n_mels, 3000)
    input_data = torch.randn(input_shape)
    
    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, input_data)

    # Convert to Core ML
    coreml_model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="logmel_data", shape=input_shape)],
        outputs=[ct.TensorType(name="output")],
        compute_units=ct.ComputeUnit.ALL,
    )

    if quantize:
        print("Quantizing encoder to 16-bit...")
        coreml_model = quantize_weights(coreml_model, nbits=16)

    return coreml_model


def convert_decoder(hparams: ModelDimensions, model: nn.Module, quantize: bool = False):
    """Convert text decoder to Core ML format."""
    print("Converting decoder...")
    model.eval()

    # Create sample inputs
    tokens_shape = (1, 1)
    audio_shape = (1, hparams.n_audio_ctx, hparams.n_audio_state)
    
    token_data = torch.randint(0, hparams.n_vocab, tokens_shape).long()
    audio_data = torch.randn(audio_shape)

    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, (token_data, audio_data))

    # Convert to Core ML
    coreml_model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="token_data", shape=tokens_shape, dtype=int),
            ct.TensorType(name="audio_data", shape=audio_shape)
        ],
        outputs=[ct.TensorType(name="output")],
        compute_units=ct.ComputeUnit.ALL,
    )

    if quantize:
        print("Quantizing decoder to 16-bit...")
        coreml_model = quantize_weights(coreml_model, nbits=16)

    return coreml_model


def main():
    parser = argparse.ArgumentParser(description="Convert Whisper models to Core ML")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", 
                "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large-v3-turbo"],
        help="Whisper model to convert"
    )
    parser.add_argument(
        "--encoder-only", 
        action="store_true",
        help="Only convert the encoder"
    )
    parser.add_argument(
        "--quantize", 
        action="store_true",
        help="Quantize weights to 16-bit for smaller model size"
    )
    parser.add_argument(
        "--optimize-ane", 
        action="store_true",
        help="Optimize for Apple Neural Engine"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for converted models"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Check ANE optimization availability
    if args.optimize_ane and not ANE_AVAILABLE:
        print("Warning: ANE optimization requested but ane_transformers not available.")
        print("Falling back to standard conversion.")
        args.optimize_ane = False

    # Load the original Whisper model
    print(f"Loading Whisper model: {args.model}")
    whisper_model = load_model(args.model).cpu()
    hparams = whisper_model.dims
    
    print(f"Model dimensions: {hparams}")

    # Choose model variant
    if args.optimize_ane:
        print("Using ANE-optimized model...")
        whisper_ane = WhisperANE(hparams).eval()
        whisper_ane.load_state_dict(whisper_model.state_dict())
        encoder = whisper_ane.encoder
        decoder = whisper_ane.decoder
    else:
        print("Using standard model...")
        encoder = whisper_model.encoder
        decoder = whisper_model.decoder

    # Convert encoder
    encoder_path = output_dir / f"coreml-encoder-{args.model}.mlpackage"
    print(f"Converting encoder to {encoder_path}")
    
    try:
        encoder_model = convert_encoder(hparams, encoder, quantize=args.quantize)
        encoder_model.save(str(encoder_path))
        print(f"✓ Encoder saved to {encoder_path}")
    except Exception as e:
        print(f"✗ Error converting encoder: {e}")
        return 1

    # Convert decoder (unless encoder-only)
    if not args.encoder_only:
        decoder_path = output_dir / f"coreml-decoder-{args.model}.mlpackage"
        print(f"Converting decoder to {decoder_path}")
        
        try:
            decoder_model = convert_decoder(hparams, decoder, quantize=args.quantize)
            decoder_model.save(str(decoder_path))
            print(f"✓ Decoder saved to {decoder_path}")
        except Exception as e:
            print(f"✗ Error converting decoder: {e}")
            return 1

    print("\n🎉 Conversion completed successfully!")
    print(f"Models saved in: {output_dir}")
    
    if args.quantize:
        print("Models were quantized to 16-bit for smaller size")
    if args.optimize_ane:
        print("Models were optimized for Apple Neural Engine")

    return 0


if __name__ == "__main__":
    sys.exit(main())
