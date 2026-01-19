# flashattn-lm-from-scratch

A from-first-principles Transformer language model implementation in PyTorch, with a Triton FlashAttention-style kernel and end-to-end training and profiling utilities.
---

## Highlights

- **Transformer LM from scratch**
  - Custom modules (Linear, RMSNorm, SwiGLU, RoPE, MHA, Transformer blocks)
- **Triton FlashAttention-style attention**
  - FlashAttention2 forward pass implementation in Triton
- **Naive attention baseline**
  - Reference scaled dot product attention (`softmax(QKᵀ)V`)
- **Profiling & benchmarking**
  - NVTX ranges + Nsight Systems/Nsight Compute friendly instrumentation
  - Benchmark script for forward/backward timing

---

## Repository structure

- `custom_modules.py`  
  Core model components and reference attention implementation.
- `train_script.py` / training entrypoint  
  Training loop, logging, config loading, dataset wiring.
- `benchmark*.py`  
  Microbenchmarks for forward/backward pass timing.
- `triton_attention.py`
  Triton kernel(s) and the `TritonAttention` autograd wrapper.
- `dataset*.py`  
  Dataset/tokenization/caching utilities (e.g., TinyStories/OpenWebText).

---

## Installation

### Requirements
- Python 3.10+ recommended
- PyTorch with CUDA (for Triton kernel + GPU training)
- Triton (typically installed as a dependency of recent PyTorch builds, but may require separate install)
- `einops`, `jaxtyping`, `datasets`, `transformers`, `tqdm` (depending on which scripts you run)

### Setup
```bash
git clone https://github.com/narenmanoj/llm_scratchwork.git
cd llm_scratchwork

python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

pip install -U pip
pip install torch torchvision torchaudio
pip install tensorboard tiktoken triton einops jaxtyping tqdm transformers datasets