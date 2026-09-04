"""
A fake `engram-mcp` stdio server for integration tests.

Speaks the same newline-delimited JSON-RPC 2.0 protocol as the real
`engram-mcp` and implements the tools the integration uses:
engram_store / engram_recall / engram_get / engram_forget / engram_stats.

Run: python fake_engram_mcp.py   (one JSON-RPC message per stdin line)
"""

import json
import sys


def main() -> None:
    memories = {}  # id -> memory dict
    counter = {"n": 0}
    started = False

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        method = msg.get("method", "")
        msg_id = msg.get("id")
        params = msg.get("params") or {}

        def send(result):
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result}) + "\n")
            sys.stdout.flush()

        if method == "initialize":
            started = True
            send({
                "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-engram", "version": "0.1.0"},
            })
        elif method == "notifications/initialized":
            pass  # notification — no response
        elif method == "tools/list":
            send({"tools": [{"name": "engram_store"}, {"name": "engram_recall"},
                            {"name": "engram_get"}, {"name": "engram_forget"}, {"name": "engram_stats"}]})
        elif method == "tools/call":
            name = params.get("name", "")
            args = params.get("arguments") or {}
            if name == "engram_store":
                counter["n"] += 1
                mem = {
                    "id": f"mem-{counter['n']}",
                    "content": args.get("content", ""),
                    "type": args.get("type", "semantic"),
                    "importance": args.get("importance", "medium"),
                    "tags": args.get("tags", []),
                    "source": args.get("source", "test"),
                    "namespace": args.get("namespace", "default"),
                    "status": "active",
                    "createdAt": 0,
                }
                memories[mem["id"]] = mem
                send({"content": [{"type": "text", "text": json.dumps(mem)}]})
            elif name == "engram_recall":
                keywords = [k.lower() for k in args.get("keywords", [])]
                limit = int(args.get("limit", 5))
                hits = [
                    m for m in memories.values()
                    if any(k in m["content"].lower() for k in keywords)
                ][:limit]
                send({"content": [{"type": "text", "text": json.dumps({"memories": hits})}]})
            elif name == "engram_get":
                mem = memories.get(args.get("id", ""))
                if mem:
                    send({"content": [{"type": "text", "text": json.dumps(mem)}]})
                else:
                    send({"content": [{"type": "text", "text": json.dumps({"error": "not found"})}]})
            elif name == "engram_forget":
                removed = memories.pop(args.get("id", ""), None)
                send({"content": [{"type": "text", "text": json.dumps({"deleted": bool(removed)})}]})
            elif name == "engram_stats":
                send({"content": [{"type": "text", "text": json.dumps({"total": len(memories)})}]})
            else:
                send({"content": [{"type": "text", "text": json.dumps({"isError": True, "detail": f"unknown tool {name}"})}], "isError": True})
        elif msg_id is not None:
            send({"error": {"code": -32601, "message": f"Method not found: {method}"}})


if __name__ == "__main__":
    main()
