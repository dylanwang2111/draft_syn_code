"""Quick standalone check that the LLM provider configured in .env (Azure
OpenAI / Azure AI Foundry, or DeepSeek) is actually reachable, before relying
on it inside the full chat assistant. Reuses chat_assistant's own client
setup, so this checks exactly what the app will actually use.

Run from the repo root:
    .venv/bin/python -m backend.test_llm_connection
"""

from __future__ import annotations

from . import chat_assistant as ca


def main():
    if ca.llm_client is None:
        print("No LLM provider configured -- set DEEPSEEK_API_KEY, or the AZURE_OPENAI_* "
              "variables, in .env first (see .env.example).")
        return

    if ca.AZURE_OPENAI_DEPLOYMENT:
        print("Provider: Azure OpenAI / Azure AI Foundry")
        print(f"Endpoint: {ca.AZURE_OPENAI_ENDPOINT}")
        print(f"Deployment: {ca.LLM_MODEL}")
    else:
        print("Provider: DeepSeek")
        print(f"Model: {ca.LLM_MODEL}")

    print("Sending a one-token test message...")
    try:
        resp = ca.llm_client.chat.completions.create(
            model=ca.LLM_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly one word: pong"}],
            max_tokens=30,   # generous: reasoning-style models spend some tokens before visible content
        )
        print("Reply:", (resp.choices[0].message.content or "").strip() or "(empty, but the call succeeded)")
        print("\nSUCCESS -- the connection works, you're good to use this from the chat assistant.")
    except Exception as e:
        print("\nFAILED:", repr(e))
        print("\nA connection/timeout error (rather than an auth error like 401/403) usually "
              "means the Azure resource is network-restricted -- e.g. only reachable over your "
              "company's VPN or from inside its VNet. Check with whoever manages the resource, "
              "or try again while connected to the VPN.")


if __name__ == "__main__":
    main()
