# Deploy MiniCPM5-1B and MiniCPM5-2B with vLLM

vLLM `>=0.21` supports MiniCPM5-1B and MiniCPM5-2B natively — no custom kernels. For production-grade throughput and OpenAI-compatible chat completions, this is the recommended path.

For MiniCPM5-15B-A2.5B, see [MiniCPM5-15B-A2.5B](#minicpm5-15b-a25b).

## Install

```bash
pip install "vllm>=0.21"          # latest (CUDA 13.x driver hosts)
# pip install "vllm==0.10.1.1"    # fallback for CUDA 12.x driver hosts
```

## OpenAI-compatible server

```bash
vllm serve openbmb/MiniCPM5-2B \
    --served-model-name MiniCPM5-2B \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
    --port 8000
```

### Tuning knobs

| Flag | Default | When to change |
| --- | --- | --- |
| `--max-model-len` | `131072` (native 128K) | drop to `8192` / `32768` to free KV-cache on small GPUs |
| `--gpu-memory-utilization` | `0.85` | drop on **shared** GPUs — vLLM hard-fails if `(free / total) < value` |
| `--dtype` | `bfloat16` | `float16` for older GPUs (newer NVIDIA GPUs prefer bf16) |
| `--enforce-eager` | unset | set if CUDA graphs OOM on tiny VRAM budgets |

## Chat completions

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-2B",
        "messages": [{"role": "user", "content": "用一句话解释什么是 GQA。"}],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": true}
    }'
```

| Mode | `enable_thinking` | `temperature` | `top_p` |
| --- | --- | --- | --- |
| MiniCPM5-2B Think | `true` | 1.0 | 0.95 |
| MiniCPM5-1B Think | `true` | 0.9 | 0.95 |
| MiniCPM5-1B No-think | `false` | 0.7 | 0.95 |

## Sample run

```bash
$ curl -sS http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"MiniCPM5-2B","messages":[{"role":"user","content":"1+1=?"}],"temperature":1.0,"top_p":0.95,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":true}}'
{
  "id": "chatcmpl-...",
  "model": "MiniCPM5-2B",
  "choices": [{"message": {"role": "assistant", "content": "2"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 14, "completion_tokens": 2, "total_tokens": 16}
}
```

## Offline / batched inference

```python
from vllm import LLM, SamplingParams

llm = LLM(model="openbmb/MiniCPM5-2B", dtype="bfloat16", max_model_len=131072)
out = llm.chat(
    [[{"role": "user", "content": "用一句话解释 GQA。"}]],
    SamplingParams(temperature=1.0, top_p=0.95, max_tokens=512),
    chat_template_kwargs={"enable_thinking": True},
)
print(out[0].outputs[0].text)
```

## Tool calling (plugin)

MiniCPM5-1B and MiniCPM5-2B emit **XML-style** tool calls. The vLLM-side parser ([vllm-project/vllm#43175](https://github.com/vllm-project/vllm/pull/43175)) was merged to `main` on 2026-05-27 but is **not in any pip release yet** — `v0.22.0` (2026-05-29) was cut before that merge and the file is absent from the v0.22.0 tree.

As a bridge, this repo ships the parser at [`tool_parsers/minicpm5xml_tool_parser.py`](../../tool_parsers/minicpm5xml_tool_parser.py) (the same file as the upstream PR). Load it via vLLM's `--tool-parser-plugin`:

```bash
vllm serve openbmb/MiniCPM5-2B \
    --served-model-name MiniCPM5-2B \
    --dtype bfloat16 --max-model-len 131072 --port 8000 \
    --enable-auto-tool-choice \
    --tool-parser-plugin /path/to/MiniCPM/tool_parsers/minicpm5xml_tool_parser.py \
    --tool-call-parser minicpm5
```

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-2B",
        "messages": [{"role": "user", "content": "What is the weather in Beijing?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }],
        "tool_choice": "auto",
        "temperature": 1.0, "top_p": 0.95, "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": true}
    }'
```

Once vLLM `v0.23` (or later) is released with the parser baked in, drop `--tool-parser-plugin` and use only `--tool-call-parser minicpm5`.

## MiniCPM5-15B-A2.5B

The model preview registers `MiniCPM5MoEForCausalLM`. SGLang and vLLM both load the published asymmetric GPTQ INT4 checkpoint directly. Its `config.json` declares `quant_method: awq` and its `model.safetensors` uses AWQ GEMM packing. [vLLM model PR #69127](https://github.com/vllm-project/vllm/pull/69127)

### Prepare the preview runtime

**Test environment.** Linux x86-64, one NVIDIA H800, driver 595.45.04, Python 3.12, and vLLM `0.34.0rc1.dev897+g9c1d45c21`. The tested CUDA nightly is `vllm/vllm-openai@sha256:663d493862a5d2c3f212f28d1bf5a5fd76dff08ec76dd1ebd72a7d230621be9a`. Overlay the five Python files from `249db3bc` as shown below. Set `MODEL_PATH` to the same complete local GPTQ directory you use with SGLang.

```bash
git init vllm-minicpm5
cd vllm-minicpm5
git fetch --depth 1 https://github.com/vllm-project/vllm.git 249db3bc16a0a3de722218110e042eb9595168ce
git checkout --detach FETCH_HEAD

MODEL_PATH="/absolute/path/to/MiniCPM5-15B-A2.5B-GPTQ"
docker run --rm -it --gpus device=0 --ipc=host \
    -p 127.0.0.1:8000:8000 \
    -v "$PWD:/vllm-src:ro" -v "$MODEL_PATH:/model:ro" \
    --entrypoint /bin/bash vllm/vllm-openai@sha256:663d493862a5d2c3f212f28d1bf5a5fd76dff08ec76dd1ebd72a7d230621be9a
```

In the container, overlay the five Python files onto the image's vLLM package. The overlay keeps the image's compiled extensions and uses a separate directory for the PR files.

```bash
VLLM_PACKAGE=$(python3 -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("vllm").origin).parent)')
OVERLAY=$(mktemp -d)
cp -as "$VLLM_PACKAGE" "$OVERLAY/vllm"

for file in \
    model_executor/models/minicpm5_moe.py \
    model_executor/models/registry.py \
    transformers_utils/config.py \
    transformers_utils/configs/__init__.py \
    transformers_utils/configs/minicpm5_moe.py; do
    cp --remove-destination "/vllm-src/vllm/$file" "$OVERLAY/vllm/$file"
done

export PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
cd "$OVERLAY"
```

### Serve the published GPTQ INT4 checkpoint

Run this in the container. The mounted directory contains the fast tokenizer and chat template. vLLM reads its quantization config, selects `auto_awq` and uses Marlin for the quantized Linear and MoE layers.

```bash
vllm serve /model \
    --host 0.0.0.0 --port 8000 --served-model-name MiniCPM5-15B-A2.5B \
    --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 1024 \
    --max-num-seqs 2 --gpu-memory-utilization 0.70 --generation-config vllm \
    --linear-backend marlin --moe-backend marlin --enforce-eager
```

On one H800, this eager launch loaded the single weight file and answered two Chat Completions requests. The 1,024-token context is sufficient for this short chat check. For a Codex client, use the longer context and Responses-capable build in the [Codex guide](codex.md).

### BF16 and speculative decoding

For BF16, mount the complete BF16 checkpoint at `/model` instead. Use the directory's own tokenizer when it is compatible. The optional conversion below is for older BF16 snapshots that declare `LlamaTokenizerFast`.

```bash
MODEL_DIR=/model
TOKENIZER_DIR=/model
```

If the BF16 tokenizer still declares `LlamaTokenizerFast`, prepare a separate copy for the Transformers 5 runtime in the pinned image. The following converts the legacy class metadata and makes the serialized post-processor match the original plain-text encoding. Run it while `MODEL_DIR` still points to the BF16 snapshot.

```bash
TOKENIZER_DIR="$(mktemp -d)"
python3 - "$MODEL_DIR" "$TOKENIZER_DIR" <<'PY'
import json
import shutil
import sys
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

source, output = map(Path, sys.argv[1:])
config = json.loads((source / "tokenizer_config.json").read_text())
assert config["tokenizer_class"] == "LlamaTokenizerFast"
assert config["add_bos_token"] is False and config["add_eos_token"] is False

raw = Tokenizer.from_file(str(source / "tokenizer.json"))
fast = Tokenizer.from_file(str(source / "tokenizer.json"))
fast.post_processor = TemplateProcessing(single="$A:0", pair="$A:0 $B:1", special_tokens=[])
fast.save(str(output / "tokenizer.json"))
config.update(tokenizer_class="PreTrainedTokenizerFast", padding_side="left")
(output / "tokenizer_config.json").write_text(json.dumps(config))
for name in ("special_tokens_map.json", "added_tokens.json", "chat_template.jinja"):
    if (source / name).is_file():
        shutil.copy2(source / name, output / name)

tokenizer = AutoTokenizer.from_pretrained(output)
assert tokenizer.is_fast and tokenizer.chat_template
for text in ("The capital of France is", "请用一句话介绍北京。"):
    assert tokenizer.encode(text) == raw.encode(text, add_special_tokens=False).ids
PY
```

For a BF16 server, use the tokenizer selected above.

```bash
vllm serve "$MODEL_DIR" --tokenizer "$TOKENIZER_DIR" \
    --host 0.0.0.0 --port 8000 --served-model-name MiniCPM5-15B-A2.5B \
    --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 4096 \
    --max-num-seqs 8 --gpu-memory-utilization 0.80 --generation-config vllm
```

For BF16 with DSpark, also mount the matching draft at `/dspark` when starting the container (add `-v "$DSPARK_PATH:/dspark:ro"` with `DSPARK_PATH` set on the host). Add `--speculative-config "$SPEC_CONFIG"` to the BF16 command after setting:

```bash
SPEC_CONFIG='{"method":"dspark","model":"/dspark","num_speculative_tokens":7}'
```

The earlier INT4 DSpark comparisons below used a separate `compressed-tensors` export. The published AWQ checkpoint was independently verified with ordinary decoding.

### Chat completion

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniCPM5-15B-A2.5B",
        "messages": [{"role": "user", "content": "Explain grouped-query attention in one sentence."}],
        "temperature": 0,
        "max_tokens": 128
    }'
```

### Validation scope and limitations

- CPU unit and registry tests at `9ea15f00` reported 8 passed and 1 skipped because the checkpoint was unavailable online. Enabling EPLB (`--enable-eplb`) or ROCm AITER shared-expert fusion (`VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`) raises `NotImplementedError` on this revision.
- Earlier target-only H800 tests at `ee331803` used a prepared BF16 tokenizer. BF16 and a separate `compressed-tensors` INT4 export each completed four 48-token greedy `/v1/completions` requests with fixed input token IDs.
- GPU smoke tests ran at `6db175bb` (prior to the EPLB and ROCm fusion guard cleanup) on one H800 with TP=1, completing four 48-token greedy requests for each BF16 and `compressed-tensors` INT4 configuration with DSpark off and on. BF16 matched greedy token IDs across DSpark modes on all 4 prompts (and matched the Hugging Face SDPA first token on 4/4 prompts). INT4 used Marlin for both Linear and MoE and matched across DSpark modes on 1/4 prompts under the default compile path (4/4 in eager mode with `max_model_len=512` and prefix caching disabled). See [vLLM PR #69127](https://github.com/vllm-project/vllm/pull/69127).

### XML tool calling

The pinned nightly has the `minicpm5` parser built in. To enable auto tool choice for Chat Completions, add `--enable-auto-tool-choice --tool-call-parser minicpm5` and send the [tool request example](#tool-calling-plugin) with `MiniCPM5-15B-A2.5B` as the model name.

On H800, Codex CLI 0.171.0 completed a read-only MCP tool call using the same published GPTQ checkpoint, the model loader at `249db3bc` and the Responses changes at `1d53c277`. This test used ordinary decoding and a 16,384-token context. [vLLM Responses PR #69128](https://github.com/vllm-project/vllm/pull/69128)

See [Codex CLI tool calling](codex.md) for the server flags, Responses protocol and a read-only MCP test.
