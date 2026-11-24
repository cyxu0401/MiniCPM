# Use Codex CLI with MiniCPM5-15B-A2.5B

Codex sends tool requests to `/v1/responses`. The server streams a function call, Codex executes the tool and sends its output back with the same `call_id`. A successful test includes both the tool result and a subsequent assistant message. [Codex MCP documentation](https://developers.openai.com/codex/mcp/)

| Server | What was verified |
| --- | --- |
| [vLLM](vllm.md#minicpm5-15b-a25b) | Codex CLI 0.171.0 called a read-only MCP tool on one H800 with the published AWQ-packed GPTQ INT4 checkpoint, ordinary decoding, and the model and Responses changes below. |
| [llama.cpp](llama_cpp.md#minicpm5-15b-a25b) | Codex CLI 0.171.0 completed the same tool round trip on one H800 with a Q4_K_M GGUF. The test combined [model PR #32644](https://github.com/ggml-org/llama.cpp/pull/32644) (`66012061`) with [parser PR #32746](https://github.com/ggml-org/llama.cpp/pull/32746) (`b0b7476e`, closed without merge). Start the combined `llama-server` with `--jinja -c 16384 -np 1 --alias MiniCPM5-15B-A2.5B` and set the client port to `8080`. |
| [SGLang](sglang.md#minicpm5-15b-a25b) | The 15B model and MiniCPM5 Responses/tool-search changes have separate previews. Use `--tool-call-parser minicpm5` for Chat Completions. The parser PR covers the 1B/2B tool path. A 15B Codex round trip on SGLang remains unverified. [Model PR #52071](https://github.com/sgl-project/sglang/pull/52071) · [Responses PR #52072](https://github.com/sgl-project/sglang/pull/52072) |
| [MLX LM](mlx.md#minicpm5-15b-a25b) | The native Apple Silicon server was tested with 15B tool calls on `/v1/chat/completions` in JSON and SSE modes. This build has no `/v1/responses` endpoint for Codex. |

## Start the tested vLLM server

Use the H800 CUDA nightly from the [vLLM guide](vllm.md#prepare-the-preview-runtime). The commands below apply the five-file Responses change to the tested vLLM base with a three-way merge, then overlay five files for model loading. The image supplies the compiled extensions. [Model PR #69127](https://github.com/vllm-project/vllm/pull/69127) · [Responses PR #69128](https://github.com/vllm-project/vllm/pull/69128)

On the Docker host, point `MODEL_PATH` to the same GPTQ directory used by SGLang. The directory includes the tokenizer and chat template.

```bash
set -euo pipefail
MODEL_PATH="/absolute/path/to/MiniCPM5-15B-A2.5B-GPTQ"
SRC=$(mktemp -d)
REPO=$(mktemp -d)
git -C "$REPO" init -q
UPSTREAM=https://github.com/vllm-project/vllm.git
BASE_REV=80456e51f8ec16e88eeb918881f7a758cbd1c2fe
TOOLS_BASE=9a77b4201bc4551f7e23ce59758dcbe6410c9e10
TOOLS_REV=1d53c2776e66dd38d9ebe27cb98d37f3132fb65d
MODEL_REV=249db3bc16a0a3de722218110e042eb9595168ce
for rev in "$BASE_REV" "$TOOLS_BASE" "$TOOLS_REV" "$MODEL_REV"; do
    git -C "$REPO" fetch --depth 1 "$UPSTREAM" "$rev"
done

git -C "$REPO" read-tree "$BASE_REV"
git -C "$REPO" diff --full-index "$TOOLS_BASE" "$TOOLS_REV" -- vllm |
    git -C "$REPO" apply --3way --cached
SOURCE_TREE=$(git -C "$REPO" write-tree)
git -C "$REPO" archive "$SOURCE_TREE" vllm | tar -xf - -C "$SRC"
git -C "$REPO" archive "$MODEL_REV" \
    vllm/model_executor/models/{minicpm5_moe,registry}.py \
    vllm/transformers_utils/config.py \
    vllm/transformers_utils/configs/{__init__,minicpm5_moe}.py | tar -xf - -C "$SRC"

docker run --rm -it --gpus device=0 --ipc=host \
    -p 127.0.0.1:8000:8000 -v "$SRC:/src:ro" -v "$MODEL_PATH:/model:ro" \
    --entrypoint /bin/bash vllm/vllm-openai@sha256:663d493862a5d2c3f212f28d1bf5a5fd76dff08ec76dd1ebd72a7d230621be9a
```

In the container, link the image's installed modules and binaries into a temporary directory, then replace the Python source with the pinned versions.

```bash
VLLM_PACKAGE=$(python3 -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("vllm").origin).parent)')
OVERLAY=$(mktemp -d)
cp -as "$VLLM_PACKAGE" "$OVERLAY/vllm"
cp -a --remove-destination /src/vllm/. "$OVERLAY/vllm/"
export PYTHONPATH="$OVERLAY:$PYTHONPATH"
export VLLM_ENABLE_RESPONSES_API_STORE=1
cd "$OVERLAY"

vllm serve /model \
    --served-model-name MiniCPM5-15B-A2.5B --host 0.0.0.0 --port 8000 \
    --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 16384 \
    --max-num-seqs 2 --gpu-memory-utilization 0.80 --generation-config vllm \
    --linear-backend marlin --moe-backend marlin --enforce-eager \
    --enable-auto-tool-choice --tool-call-parser minicpm5 --reasoning-parser qwen3
```

The `minicpm5` parser converts model XML to function calls. The Responses changes also handle tool discovery and replay. `VLLM_ENABLE_RESPONSES_API_STORE=1` enables server-side history for a client that sends `store=true` and then refers to a `previous_response_id`. A client can also send the conversation and tool result in full.

## Run a read-only MCP tool

Add a named provider to `~/.codex/config.toml` on the client. Select it for the test with `-c`. For a server listening on another port, update `base_url`. [Codex configuration reference](https://developers.openai.com/codex/config-reference/)

```toml
[model_providers.minicpm_local]
name = "MiniCPM5 local"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
supports_websockets = false
```

From the root of this MiniCPM checkout, register the example [read-only MCP server](examples/minicpm5_mcp_probe.py) for one run. The tool reads only the supplied marker file. The prompt names the tool and omits the marker value. [Codex non-interactive mode](https://developers.openai.com/codex/noninteractive/)

```bash
PROBE="$PWD/docs/deployment/examples/minicpm5_mcp_probe.py"
TEST_DIR=$(mktemp -d)
printf 'CODEX_MINICPM_XML_OK\n' > "$TEST_DIR/marker.txt"
PROBE_ARGS=$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' \
    "$PROBE" "$TEST_DIR/marker.txt")
codex exec --ephemeral --skip-git-repo-check --cd "$TEST_DIR" --sandbox read-only --json \
    -m MiniCPM5-15B-A2.5B -c 'model_provider="minicpm_local"' \
    -c model_context_window=16384 -c model_auto_compact_token_limit=14000 \
    -c 'model_reasoning_effort="low"' -c 'mcp_servers.fixture.command="python3"' \
    -c "mcp_servers.fixture.args=$PROBE_ARGS" -c mcp_servers.fixture.required=true \
    -c 'mcp_servers.fixture.enabled_tools=["read_sentinel"]' \
    -c 'mcp_servers.fixture.tools.read_sentinel.approval_mode="approve"' \
    'Use the read_sentinel tool from the fixture MCP server. Reply only with the tool result.'
```

Check the JSON events. An `item.completed` event should contain `item.type="mcp_tool_call"`, `tool="read_sentinel"`, `status="completed"` and the marker in `result.content`. A later `agent_message` should repeat `CODEX_MINICPM_XML_OK`. If the server returns HTTP 404, check `/v1/responses` on the configured port. If XML appears in assistant text, check the parser revision and launch flags.
