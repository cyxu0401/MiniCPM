"""A stdio MCP server with one read-only fixture tool for CLI verification."""

import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
tools = [{
    "name": "read_sentinel",
    "description": "Read the fixed sentinel value for this smoke test.",
    "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
}]

for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if "id" not in request:
        continue
    response = {"jsonrpc": "2.0", "id": request["id"]}
    if method == "initialize":
        response["result"] = {
            "protocolVersion": request["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fixture", "version": "1.0.0"},
        }
    elif method == "tools/list":
        response["result"] = {"tools": tools}
    elif method == "tools/call" and request["params"]["name"] == "read_sentinel":
        if request["params"].get("arguments"):
            response["error"] = {"code": -32602, "message": "read_sentinel takes no arguments"}
        else:
            value = source.read_text().strip()
            response["result"] = {"content": [{"type": "text", "text": value}]}
    elif method == "ping":
        response["result"] = {}
    else:
        response["error"] = {"code": -32601, "message": "Unknown method"}
    print(json.dumps(response, separators=(",", ":")), flush=True)
