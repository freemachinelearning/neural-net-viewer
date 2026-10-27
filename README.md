# Model Viewer

A React app for exploring PyTorch models: browse the layer hierarchy, see the
architecture as a block diagram, and inspect every weight tensor.

Files are parsed **entirely in the browser**. `.pt`/`.pth` files are zip archives
containing a pickle plus raw tensor storage; this app includes a small pickle
interpreter (`src/lib/pickle.ts`) that reads them without running Python.

```bash
npm install
npm run dev        # http://localhost:5173
```

Click **Try a demo CNN** to explore without a file, or generate real samples with
`python scripts/make_sample_models.py` (needs PyTorch).

## Supported files

| What you saved                          | What you get                                             |
| --------------------------------------- | -------------------------------------------------------- |
| `torch.save(model)`                     | Exact module classes and hyperparameters (in_features, kernel_size, …) |
| `torch.save(model.state_dict())`        | Hierarchy from parameter names; layer types inferred from shapes |
| Checkpoint dicts (`{"model_state_dict": …}`) | Pick which entry to view from the header dropdown |
| TorchScript (`torch.jit.save`)          | Module tree and weights                                  |
| `.safetensors`                          | Same as a state_dict                                     |

Supported dtypes: float64/32/16, bfloat16, int64/32/16/8, uint8, bool.
Not supported: legacy pre-1.6 (non-zip) `torch.save` files, quantized and sparse tensors.

## Weight views

- **Filters**: conv kernels as a grid (output filter × input channel); first layers with 3 input channels can be shown as RGB images.
- **Neurons**: a Linear layer drawn as input → output neurons, with edge opacity = |weight|. Big layers show the neurons with the largest L2 norm.
- **Heatmap**: any tensor as a matrix (dim 0 × rest), downsampled by max-|w| per block.
- **Distribution**: histogram with a linear/log toggle.
- **Values**: paged table of raw numbers.

Note that the diagram shows modules in the order they were registered in
`__init__`, which is usually but not always the order `forward()` runs them in.
