---
name: minicpm5-deploy-mlx
description: Run MiniCPM5-1B or MiniCPM5-2B natively on Apple Silicon with Apple's MLX framework. Use when the user has an Apple Silicon Mac and asks for "MLX", "mlx_lm", "mlx_lm.convert", "mlx_lm.generate", or wants the fastest path on Apple Silicon.
  Also describes the MLX LM runtime requirements for MiniCPM5-15B-A2.5B affine INT4 inference, DSpark, and XML4 tool calling.
---

# Deploy MiniCPM5-1B and MiniCPM5-2B with MLX (Apple Silicon)

Apple's on-device tensor framework. Highest throughput on M-series. Stays inside one Python process — no separate server, no `llama.cpp` build chain.

For MiniCPM5-15B-A2.5B, start at [MiniCPM5-15B-A2.5B](#minicpm5-15b-a25b). For XML4 tools, use [MiniCPM5 XML4 tool calling](#minicpm5-xml4-tool-calling).

## Required input

| Var | Example | Default |
| --- | --- | --- |
| `MLX_REPO` | `openbmb/MiniCPM5-2B-MLX` (pre-converted 4-bit affine) | required; `openbmb/MiniCPM5-1B-MLX` also works |
| OR `HF_REPO` + `QUANT` | `openbmb/MiniCPM5-2B`, `4bit` or `bf16` | for local conversion |
| `MAX_TOKENS` | `200` | `200` |

## Steps

### 1. Install (once)

```bash
pip install "mlx-lm>=0.31" "gguf"
```

### 2A. Use a pre-converted MLX repo (recommended when available)

```bash
mlx_lm.generate --model "${MLX_REPO}" \
    --prompt "<|im_start|>user
1+1=?<|im_end|>
<|im_start|>assistant
" \
    --max-tokens ${MAX_TOKENS} --temp 1.0 --top-p 0.95
```

### 2B. Convert from a HF checkpoint locally (advanced)

Use `mlx_lm.convert` only if you have a self-trained HF fp16 checkpoint:

```bash
HF=/path/to/your-fp16-hf

# Convert: bf16
mlx_lm.convert --hf-path "$HF" --mlx-path ./minicpm5-mlx-bf16

# Convert: 4-bit (smaller / faster)
mlx_lm.convert --hf-path "$HF" --mlx-path ./minicpm5-mlx-q4 -q --q-bits 4
```

Then run as in 2A.

### 3. Validate

The reply should contain `"2"` for `1+1=?`.

## OpenAI-compatible server (mlx-lm)

```bash
mlx_lm.server --model "${MLX_REPO}" --host 127.0.0.1 --port 8000

curl http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "messages": [{"role":"user","content":"1+1=?"}],
        "temperature": 1.0, "top_p": 0.95, "max_tokens": 64
    }'
```

## Common pitfalls

- **Slow first generate**: MLX JIT-compiles kernels on first call (~5-10 s); subsequent calls hit the warm cache.
- **Model runs past `<|im_end|>`**: only happens on `mlx-lm < 0.31` (older versions ignore multi-id `eos_token_id` lists). Upgrade, or pass `--extra-eos-token "<|im_end|>"` as a manual override — `<|im_end|>` is token id 130073 and is already listed in `generation_config.json` on 0.31+.

## When NOT to use

- Not on Apple Silicon → `minicpm5-deploy-llama-cpp` (CPU/CUDA) or `minicpm5-deploy-vllm` (CUDA)
- Want a desktop GUI → `minicpm5-deploy-lmstudio` (LM Studio bundles an MLX runtime)
- Want one-line CLI run → `minicpm5-deploy-ollama`

## MiniCPM5-15B-A2.5B

Run natively on Apple Silicon with Metal in a Python virtual environment. The tested system used an M5 with 16 GiB, macOS 26.6.2 arm64, Python 3.13.15, MLX 0.32.4 and Transformers 5.23.0. [Installation and model source](../../docs/deployment/mlx.md#minicpm5-15b-a25b)

The native MLX server was tested with Chat Completions. Use a Responses-capable server for the Codex CLI example. [Codex CLI tool calling](../../docs/deployment/codex.md)

1. On Apple Silicon, resolve `MODEL_DIR` to a prepared MLX checkpoint directory. The tested format is affine INT4 with group size 128.
2. For deployment, select an official MLX LM release that includes `minicpm5_moe`. Upstream v0.32.2 lacks this model. For local validation, use the cookbook's environment and run the short `mlx_lm generate` example to check the target. The model and DSpark server are tracked in the fork. [15B launch commands](../../docs/deployment/mlx.md#minicpm5-15b-a25b) · [cyxu0401/mlx-lm#1](https://github.com/cyxu0401/mlx-lm/pull/1)
3. Send a short `/v1/chat/completions` request using `default_model`. For DSpark, confirm that `python -m mlx_lm server --help` lists `--speculative-algorithm`. Restart with the matching 15B draft and unquantized KV caches, then check the server log for proposed and accepted tokens. Both modes ran on an Apple M5 with 16 GiB of unified memory.

## MiniCPM5 XML4 tool calling

1. Check `python -m mlx_lm server --help` for `--tool-call-parser`. The XML4 parser is tracked in the fork and can serve the published 1B and 2B MLX checkpoints directly. The tested 15B configuration used this parser with the `minicpm5_moe` model package from PR #1 and ordinary decoding. [XML4 launch commands](../../docs/deployment/mlx.md#minicpm5-xml4-tool-calling) · [cyxu0401/mlx-lm#2](https://github.com/cyxu0401/mlx-lm/pull/2)
2. Start the server with `--tool-call-parser minicpm5` and `has_tool_sep=false` in `--chat-template-args`. Check the function name, arguments, and call ID in `message.tool_calls` (or assemble `delta.tool_calls` when streaming). Send the result back with that ID in `tool_call_id` and confirm the next turn uses it.

## Reference

[`docs/deployment/mlx.md`](../../docs/deployment/mlx.md)
