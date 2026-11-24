# Deploy MiniCPM5-1B and MiniCPM5-2B with MLX (Apple Silicon)

[MLX](https://github.com/ml-explore/mlx) is Apple's on-device tensor framework. For MiniCPM5-1B and MiniCPM5-2B it is the recommended path on Apple Silicon (M1–M4) when you want **highest throughput** and want to stay inside one Python process — no separate server, no `llama.cpp` build chain.

For MiniCPM5-15B-A2.5B, see [MiniCPM5-15B-A2.5B](#minicpm5-15b-a25b). For the XML4 parser requirements on 1B, 2B, or 15B, see [MiniCPM5 XML4 tool calling](#minicpm5-xml4-tool-calling).

## TL;DR

```bash
pip install "mlx-lm>=0.31"

# Run the official pre-converted MLX repo directly
# (config.json declares "quantization": {"bits": 4, "mode": "affine"})
mlx_lm.generate --model openbmb/MiniCPM5-2B-MLX \
    --prompt "<|im_start|>user
1+1=?<|im_end|>
<|im_start|>assistant
" \
    --max-tokens 200 --temp 1.0 --top-p 0.95
```

## Building MLX weights from your own checkpoint (advanced)

If you have a self-trained HF fp16 checkpoint and want to produce MLX weights, use `mlx_lm.convert`:

```bash
HF=/path/to/your-fp16-hf

# bf16 master copy
mlx_lm.convert --hf-path "$HF" --mlx-path ./minicpm5-mlx-bf16

# 4-bit (smaller / faster, ~4.5 bits/weight on average)
mlx_lm.convert --hf-path "$HF" --mlx-path ./minicpm5-mlx-q4 -q --q-bits 4
```

The 4-bit pass logs `[INFO] Quantized model with 4.501 bits per weight.` — the slight overshoot above 4 bits is from keeping `embed_tokens` and `lm_head` in higher precision, which preserves quality on the small, untied vocabulary head.

## Inference

### One-shot generate

```bash
mlx_lm.generate --model ./minicpm5-mlx-q4 \
    --prompt "<|im_start|>user
鸡兔同笼，头共10个，脚共28只，问鸡和兔各几只？<|im_end|>
<|im_start|>assistant
" \
    --max-tokens 200 --temp 1.0 --top-p 0.95
```

```text
首先，理解问题：总共有10个头（鸡和兔都是头），总共有28只脚。我们需要找出鸡和兔各自的数量。

设鸡的数量为x，兔的数量为y。那么：
1. 头数：总头数 = 鸡的数量 + 兔的数量 = x + y = 10。
2. 脚数：鸡有2只脚，兔有4只脚，总脚数 = 2x + 4y = 28。
…
```

### Python API (streaming)

```python
from mlx_lm import load, stream_generate

model, tk = load("./minicpm5-mlx-q4")

prompt = (
    "<|im_start|>user\n"
    "用一句话解释什么是 GQA。<|im_end|>\n"
    "<|im_start|>assistant\n"
)

for resp in stream_generate(
    model, tk, prompt=prompt, max_tokens=512,
    sampler=None,                    # default temp/top_p
):
    print(resp.text, end="", flush=True)
print()
```

> ℹ️ `<|im_end|>` is **token id 130073** in this tokenizer, and `generation_config.json` already lists it in `eos_token_id: [1, 130073]`. mlx-lm reads that list and adds both ids to the stop set, so no `--extra-eos-token` flag is needed — the model stops at end-of-turn on its own.

## Recommended sampling

| Mode | `--temp` | `--top-p` | When to use |
| --- | --- | --- | --- |
| MiniCPM5-2B Think | 1.0 | 0.95 | reasoning, math, code, multi-step (model auto-emits `<think>` block) |
| MiniCPM5-1B Think | 0.9 | 0.95 | reasoning, math, code, multi-step (model auto-emits `<think>` block) |
| MiniCPM5-1B No-think | 0.7 | 0.95 | fast assistant, latency-bound |

For MiniCPM5-1B, both modes are activated by sampling parameters only — the released chat template auto-injects `<think>\n` when no `system` message disables it, so you get think-mode behaviour by default.

## Q&A

### Model never stops generating

Confirm you're on `mlx-lm >= 0.31`. Older versions ignored multi-id `eos_token_id` lists in `generation_config.json` and would only stop at `</s>` (id 1) — on a chat-template turn the model never emits `</s>` so it ran past the turn boundary. 0.31+ honours the full list (`[1, 130073]`) and stops at `<|im_end|>` automatically. As a last-resort override you can still pass `--extra-eos-token "<|im_end|>"`.

## See also

- [`transformers.md`](./transformers.md) — same checkpoint, CPU / CUDA path
- [`llama_cpp.md`](./llama_cpp.md) — alternative on-device path (CPU + Metal)
- [`lmstudio.md`](./lmstudio.md) — desktop GUI consumer of the same MLX models

## MiniCPM5-15B-A2.5B

Use an Apple Silicon Mac with Python 3.11 or newer. Deployment requires an official MLX LM release that includes `minicpm5_moe`. As of 2026-11-24, the latest upstream release is v0.32.2 and lacks this model. The native model and DSpark server used in the examples are tracked in the `cyxu0401/mlx-lm` fork. [Upstream release](https://github.com/ml-explore/mlx-lm/releases/tag/v0.32.2) · [Model and DSpark PR #1](https://github.com/cyxu0401/mlx-lm/pull/1)

Set `MODEL_DIR` to a prepared MLX checkpoint directory with its tokenizer and chat template. The M5 run used a single-file target with 4-bit affine weights, repacked from asymmetric GPTQ with group size 128.

To reproduce that run locally, start in a checkout containing the model implementation at `1c4e747`. The test used Python 3.13.15 with the following packages.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e . 'mlx==0.32.4' 'transformers==5.23.0'
python -c "from mlx_lm.models.minicpm5_moe import Model"
```

### Generate a short completion

The model PR used this command to check the native model package on Linux CPU. It also works as a local generation check before starting the server.

```bash
MODEL_DIR="/path/to/MiniCPM5-15B-A2.5B-MLX"
python -m mlx_lm generate --model "$MODEL_DIR" \
    --prompt "The secret to baking a good cake is" \
    --max-tokens 24 --temp 0 --chat-template-config '{"enable_thinking":false}'
```

### Start the server

```bash
MODEL_DIR="/path/to/MiniCPM5-15B-A2.5B-MLX"
python -m mlx_lm server --model "$MODEL_DIR" \
    --host 127.0.0.1 --port 8080 --prefill-step-size 128 \
    --allocator-cache-limit 268435456
```

### Speculative decoding with DSpark

Use a build with DSpark server support. Stop the baseline server and select the matching DSpark draft for MiniCPM5-15B-A2.5B. The command uses unquantized KV caches.

```bash
DSPARK_DIR="/path/to/MiniCPM5-15B-A2.5B-DSpark"
python -m mlx_lm server --model "$MODEL_DIR" \
    --host 127.0.0.1 --port 8080 --prefill-step-size 128 \
    --allocator-cache-limit 268435456 --speculative-algorithm DSPARK \
    --speculative-draft-model-path "$DSPARK_DIR"
```

### Chat completion

In another terminal, check `/v1/models` and send a short chat request. `default_model` selects the checkpoint passed to the server.

```bash
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default_model",
        "messages": [{"role": "user", "content": "Write the integers from 1 through 12 in ascending order, separated by a comma and a space. Reply with just the list."}],
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": false}
    }'
```

On an Apple M5 with 16 GiB of unified memory, this prompt returned the same list with ordinary decoding and with DSpark at `1c4e747`. Both responses used 35 completion tokens. DSpark accepted 29 of the 35 proposed tokens in that request.

## MiniCPM5 XML4 tool calling

The XML4 parser reads the model's XML tool calls and returns structured `tool_calls` in both JSON and streaming chat completions. It uses the tokenizer's reasoning markers to keep reasoning text separate. Enable it with `--tool-call-parser minicpm5`. Upstream v0.32.2 lacks this option. [XML4 parser PR #2 in the cyxu0401/mlx-lm fork](https://github.com/cyxu0401/mlx-lm/pull/2)

The published 1B and 2B MLX checkpoints use the parser build directly. The MiniCPM5-15B-A2.5B test used the server and parser at `7b89628` with the `mlx_lm/models/minicpm5_moe` package from the model PR at `1c4e747`. It used ordinary decoding.

### Start the XML4 server

Check that `python -m mlx_lm server --help` lists `--tool-call-parser`, then select the checkpoint.

```bash
MODEL_PATH="openbmb/MiniCPM5-2B-MLX"
# For 1B, set MODEL_PATH="openbmb/MiniCPM5-1B-MLX".
# For 15B, set MODEL_PATH="/path/to/MiniCPM5-15B-A2.5B-MLX".
python -m mlx_lm server --model "$MODEL_PATH" --tool-call-parser minicpm5 \
    --chat-template-args '{"has_tool_sep":false,"enable_thinking":false}' \
    --host 127.0.0.1 --port 8080 --prefill-step-size 128 --decode-concurrency 1
```

### Send a tool request

```bash
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default_model",
        "messages": [{"role": "user", "content": "Use lookup_temperature to check the temperature in Beijing."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup_temperature",
                "description": "Look up the temperature for a city in Celsius.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 128
    }'
```

Check `choices[0].message.tool_calls` for the function name, JSON arguments, and call ID. To continue the conversation, send the assistant's tool calls back and add a `role: "tool"` message with the matching `tool_call_id` for each result. To test SSE, set `"stream": true` and use `curl -N`. Assemble the fragments in `choices[0].delta.tool_calls` by call index before returning the results.

On Apple M5, the 1B and 2B affine INT4 checkpoints (group size 64) each completed two tool-call and result rounds in each transport with thinking disabled. The 15B affine INT4 checkpoint (group size 128) completed four three-turn conversations covering JSON and SSE with thinking enabled and disabled. All 12 requests in the 15B test returned HTTP 200.
