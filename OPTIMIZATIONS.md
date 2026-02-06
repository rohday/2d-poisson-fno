# Optimizations

Model efficiency improvements for edge deployment.

## 1. Depthwise Separable Spectral Convolution

Standard spectral conv applies dense mixing across all channels at each frequency:
```
out[c_out] = sum over c_in of (weights[c_in, c_out] * fft(x[c_in]))
```

Depthwise version filters each channel independently, then relies on the 1×1 conv (W layer) for channel mixing:
```
out[c] = weights[c] * fft(x[c])  →  then W mixes channels
```

| Metric | Full | Depthwise | Reduction |
|--------|------|-----------|-----------|
| Parameters | 1,188,321 | 45,537 | 26x |
| VRAM | 25 MB | 8 MB | 3x |

## 2. Spectral Truncation

FFT produces N/2 frequency modes per dimension. Most energy is in low frequencies.  
We keep only `modes` lowest frequencies and discard the rest.

- modes=12: keeps 12/32 = 37.5% of spectrum (default)
- modes=16: keeps 16/32 = 50% of spectrum (slightly better accuracy)

Higher modes = more capacity but more parameters.

## Model Variants

| Model | Config | Params | Accuracy |
|-------|--------|--------|----------|
| z8k_1 (unoptimised)| w=32, d=4, m=12 | 1.19M | 95.7% |
| d6w48m12 | w=48, d=6, m=12 | 104k | 94.0% |
| d6w48m16 | w=48, d=6, m=16 | 168k | 94.1% |

d6w48m16 could've been more accurate but training was stopped early accidentally.
