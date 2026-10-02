"""
Tests for the WebSocket transport (v0.8.1 hardening).

Covers request/reply round-trips, the reply timeout on silent servers, and
malformed-frame tolerance on the server side.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import websockets

from agentlink.protocol.message import AgentAddress, AgentMessage, MessageType
from agentlink.transport import WSTransport

PLANNER = AgentAddress("planner", "default")
WORKER = AgentAddress("worker", "default")


def make_message(content="hello"):
    return AgentMessage(
        type=MessageType.REQUEST,
        sender=PLANNER,
        recipient=WORKER,
        content=content,
    )


async def _start_server(on_message=None, reply_timeout=30.0):
    server = WSTransport(
        host="127.0.0.1", port=0, on_message=on_message, reply_timeout=reply_timeout
    )
    await server.start()
    port = server._server.sockets[0].getsockname()[1]
    return server, port


async def test_roundtrip_with_reply():
    async def echo(msg):
        return msg.reply(f"echo: {msg.content}")

    server, port = await _start_server(on_message=echo)
    try:
        client = WSTransport()
        await client.connect(f"ws://127.0.0.1:{port}")
        try:
            reply = await client.send(make_message("ping"))
            assert reply is not None
            assert reply.type is MessageType.REPLY
            assert reply.content == "echo: ping"
        finally:
            await client.stop()
    finally:
        await server.stop()


async def test_silent_server_times_out():
    # Default server (no on_message) never replies; the client must not hang.
    server, port = await _start_server(reply_timeout=0.5)
    try:
        client = WSTransport(reply_timeout=0.5)
        await client.connect(f"ws://127.0.0.1:{port}")
        try:
            with pytest.raises(asyncio.TimeoutError):
                await client.send(make_message())
        finally:
            await client.stop()
    finally:
        await server.stop()


async def test_malformed_frame_gets_error_and_handler_survives():
    async def echo(msg):
        return msg.reply("ok")

    server, port = await _start_server(on_message=echo)
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send("this is not json")
            reply = json.loads(await ws.recv())
            assert reply["type"] == MessageType.ERROR.value
            assert reply["content"]["error"] == "malformed message frame"

            # The handler must still be alive for well-formed frames.
            await ws.send(json.dumps(make_message("after").to_dict()))
            reply = json.loads(await ws.recv())
            assert reply["type"] == MessageType.REPLY.value
            assert reply["content"] == "ok"
    finally:
        await server.stop()


async def test_send_without_connect_raises():
    client = WSTransport()
    with pytest.raises(RuntimeError):
        await client.send(make_message())
