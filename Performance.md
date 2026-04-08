# Performace of various models

| Model | Channels          | Head      | # Params | Seq Lenght | Best Val AUC | Easy AUC | Easy Acc | Hard AUC | Hard Acc | Efficiency (Hard AUC / 10k params) | Notes                              |
|-------|-------------------|-----------|----------|------------|--------------|----------|----------|----------|----------|----------------------------|------------------------------------|
| v1    | 1→16→32→64→64→96  | 96→32→1   | 168417   | 17 ep    | 0.9999       | 0.9999   | 0.9968   | 0.9662   | 0.9547   | 0.574                        | Baseline                           |
| v2    | 1→8→16→32→32→64   | 64→32→1   | 51393    | 25 ep    | 1.0000       | 0.9999   | 0.9959   | 0.9625   | 0.9551   | 0.187                        | Compression of layer dimensions     |
| v3    | 1→8→16→32→32→64   | 64→1      | 49345    | 34 ep    | 0.9982       | 0.9979   | 0.9818   | 0.9414   | 0.9400   | 0.191                        | Flattened head, too much compression|
| v4    | 1→8→16→32→32→64   | 64→16→1   | 50337    | 12 ep    | 0.9997       | 0.9994   | 0.9876   | 0.9473   | 0.9512   | 0.188                        | Reduction of head                  |
| v5    | 1→8→16→24→24→32   | 32→16→1   | 24145    | 34 ep    | 0.9999       | 0.9998   | 0.9957   | 0.9542   | 0.9503   | 0.395                        | Compression of layers (best tradeoff) |
| v6    | 1→8→16→24→32      | 32→16→1   | 17737    | 35 ep    | 0.9985       | 0.9981   | 0.9836   | 0.9347   | 0.9418   | 0.523                         | Too shallow, performance drop |

**Training Configuration**
- Epochs: 50 (early stopping, patience = 5)
- Train size: 30000
- Validation size: 6000
- Batch size: 64
- Optimizer: AdamW
- Loss: BCEWithLogitsLoss
- Seed: 42

---