# Installing GPU-Enabled PyTorch on Windows

## Overview

This guide provides step-by-step instructions for setting up PyTorch with GPU acceleration on Windows 11/10. Unlike TensorFlow, PyTorch has **excellent native Windows GPU support** - no WSL2 required!

## Prerequisites

### Hardware Requirements

- **NVIDIA GPU** with Compute Capability ≥ 7.0
  - Recommended: RTX 20xx series or newer (RTX 3050, 3060, 3070, etc.)
  - Minimum: 8 GB VRAM for training deep learning models
  - Your GPU: **GeForce RTX 3050** ✅ (Compatible)

### Software Requirements

- **Windows 10/11** (64-bit)
- **Python 3.10** (as specified in project requirements)
- **NVIDIA GPU Driver** ≥ 560.xx series (November 2025)
  - Check your driver: `nvidia-smi` in Command Prompt
  - Update if needed: https://www.nvidia.com/Download/index.aspx

## Installation Steps

### Step 1: Verify NVIDIA Driver

1. Open Command Prompt or PowerShell
2. Run: `nvidia-smi`
3. You should see:
   - GPU name (e.g., "GeForce RTX 3050")
   - Driver version (should be ≥ 560.xx)
   - CUDA version (e.g., "CUDA Version: 12.4")

**If `nvidia-smi` is not found:**
- Install/update NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx
- Restart your computer after installation

### Step 2: Install PyTorch with CUDA 12.4

**Option A: Using pip (Recommended)**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

This installs:
- PyTorch with CUDA 12.4 support
- torchvision (for image transforms)
- torchaudio (for audio processing)

**Option B: CPU-only (if no GPU available)**

```bash
pip install torch torchvision torchaudio
```

### Step 3: Verify Installation

Test GPU in Python:

```python
import torch

# Check CUDA availability
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
```

**Expected Output:**
```
CUDA Available: True
GPU Name: NVIDIA GeForce RTX 3050
CUDA Version: 12.4
```

### Step 4: Quick Test

Test GPU in Python:

```python
import torch

# Check CUDA availability
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Test GPU operation
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = x @ y  # Matrix multiplication on GPU
    print(f"GPU test successful! Result on: {z.device}")
```

## Advantages Over TensorFlow on Windows

✅ **Native Windows Support** - No WSL2 required
✅ **Simpler Installation** - Single pip command
✅ **Better GPU Utilization** - Optimized CUDA operations
✅ **Easier Debugging** - Eager execution by default
✅ **Active Development** - Regular updates and improvements

## Troubleshooting

### Issue: `torch.cuda.is_available()` returns `False`

**Possible Causes:**
1. **PyTorch installed without CUDA**
   - Solution: Reinstall with CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`

2. **NVIDIA driver not installed/outdated**
   - Solution: Install/update drivers from NVIDIA website

3. **CUDA version mismatch**
   - Solution: Ensure driver supports CUDA 12.4 (check with `nvidia-smi`)

### Issue: "CUDA out of memory"

**Solutions:**
- Reduce batch size in training (e.g., from 32 to 16)
- Use mixed precision training (already implemented in code)
- Close other GPU-intensive applications
- Reduce model size if possible

### Issue: "Could not load cudnn_cnn_infer64_8.dll"

**Solution:**
- Reinstall PyTorch with CUDA: `pip uninstall torch torchvision torchaudio && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`

### Issue: Slow training despite GPU

**Check:**
1. Verify GPU is being used: `nvidia-smi` during training
2. Check DataLoader `num_workers` (should be > 0 for GPU)
3. Ensure `pin_memory=True` in DataLoader
4. Verify mixed precision is enabled (check training logs)

## Performance Expectations

With GPU acceleration, you should see:
- **10-50× speedup** compared to CPU training
- **Training time**: ~30-60 minutes for CNN/LSTM models (vs 2-4 hours on CPU)
- **GPU utilization**: 80-100% during training (check with `nvidia-smi`)

## Monitoring GPU Usage

During training, open a new terminal and run:

```bash
nvidia-smi -l 1
```

This shows real-time GPU utilization, memory usage, and temperature.

## Next Steps

1. ✅ Verify GPU setup: `python -c "import torch; print('CUDA:', torch.cuda.is_available())"`
2. ✅ Test setup: `python test_setup.py`
3. ✅ Train models: `python main.py --mode train`

## Additional Resources

- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
- **CUDA Installation Guide**: https://developer.nvidia.com/cuda-downloads
- **PyTorch Installation**: https://pytorch.org/get-started/locally/

## Current Setup Status

- ✅ NVIDIA GPU detected: GeForce RTX 3050
- ✅ NVIDIA Drivers: Version 581.80
- ✅ CUDA Version: 13.0 (compatible with PyTorch CUDA 12.4)
- ✅ PyTorch: Install with `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`

## Notes

- PyTorch CUDA 12.4 is compatible with CUDA 13.0 drivers (backward compatible)
- No manual CUDA Toolkit installation needed - PyTorch includes everything
- No cuDNN installation needed - PyTorch includes cuDNN 9.x
- All dependencies are bundled with PyTorch installation

