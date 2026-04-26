"""Run the OpenAI-compatible HTTP shim.

The shim accepts standard OpenAI chat-completion requests and routes each one
to the provider with the best capability match for the task. Any OpenAI
client SDK (or curl, Cursor, Continue, etc.) can point ``base_url`` at this
server and get capability-aware routing transparently.

Defaults to stub mode (no API keys, fake responses) so the example runs
out of the box. Pass ``--real`` to use real provider SDKs.

Example client::

    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
    resp = client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "Refactor this login helper"}],
    )
    print(resp.choices[0].message.content)
    # The router's decision is on resp.x_frontier_router (when the client
    # surfaces unknown fields), or visible via the /route endpoint.
"""

from __future__ import annotations

from frontier_router.server import serve

if __name__ == "__main__":
    serve(host="0.0.0.0", port=8080, mode="stub")
