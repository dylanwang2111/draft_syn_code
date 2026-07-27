"""Quick standalone check that the LLM provider configured in .env (Azure
OpenAI / Azure AI Foundry, a direct OpenAI key, or DeepSeek) is actually
reachable, before relying on it inside the full chat assistant. Reuses
chat_assistant's own client setup, so this checks exactly what the app will
actually use.

Run from the repo root:
    .venv/bin/python -m backend.test_llm_connection
"""

from __future__ import annotations

from openai import AuthenticationError, APIConnectionError, APITimeoutError, PermissionDeniedError, RateLimitError

from . import chat_assistant as ca


def main():
    if ca.llm_client is None:
        print("No LLM provider configured -- set DEEPSEEK_API_KEY, OPENAI_API_KEY, or the "
              "AZURE_OPENAI_* variables, in .env first (see .env.example).")
        return

    if ca.AZURE_OPENAI_API_KEY and ca.AZURE_OPENAI_ENDPOINT and ca.AZURE_OPENAI_DEPLOYMENT:
        print("Provider: Azure OpenAI / Azure AI Foundry")
        print(f"Endpoint: {ca.AZURE_OPENAI_ENDPOINT}")
        print(f"Deployment: {ca.LLM_MODEL}")
    elif ca.OPENAI_API_KEY:
        print("Provider: OpenAI (direct, platform.openai.com)")
        print(f"Model: {ca.LLM_MODEL}")
    else:
        print("Provider: DeepSeek")
        print(f"Model: {ca.LLM_MODEL}")

    print("Sending a one-token test message...")
    try:
        # newer reasoning-family models (o1/o3/gpt-5.x) reject max_tokens and
        # want max_completion_tokens instead -- try the older/more widely
        # supported name first, fall back on the one unsupported_parameter
        # error this specific mismatch produces
        kwargs = dict(model=ca.LLM_MODEL,
                      messages=[{"role": "user", "content": "Reply with exactly one word: pong"}])
        try:
            resp = ca.llm_client.chat.completions.create(max_tokens=30, **kwargs)
        except Exception as e:
            if "max_completion_tokens" in str(e):
                resp = ca.llm_client.chat.completions.create(max_completion_tokens=30, **kwargs)
            else:
                raise
        print("Reply:", (resp.choices[0].message.content or "").strip() or "(empty, but the call succeeded)")
        print("\nSUCCESS -- the connection works, you're good to use this from the chat assistant.")
    except RateLimitError as e:
        print("\nFAILED:", repr(e))
        if "insufficient_quota" in str(e):
            print("\nThe key itself is valid (this got past auth), but the account/project it "
                  "belongs to has no billing or quota set up. Check "
                  "https://platform.openai.com/settings/organization/billing for OpenAI, or the "
                  "Azure resource's quota page for Azure.")
        else:
            print("\nRate-limited -- the key works, you're just sending requests faster than the "
                  "account's current limit allows. Wait a bit and try again.")
    except (AuthenticationError, PermissionDeniedError) as e:
        print("\nFAILED:", repr(e))
        print("\nThe key was rejected (401/403) -- double check it was copied in full and hasn't "
              "been revoked/rotated, and that it matches the provider you configured (an Azure "
              "resource key and a platform.openai.com key are not interchangeable).")
    except (APIConnectionError, APITimeoutError) as e:
        print("\nFAILED:", repr(e))
        print("\nCouldn't reach the endpoint at all. For Azure this usually means the resource is "
              "network-restricted -- e.g. only reachable over your company's VPN or from inside "
              "its VNet. Check with whoever manages the resource, or try again while connected to "
              "the VPN.")
    except Exception as e:
        print("\nFAILED:", repr(e))


if __name__ == "__main__":
    main()
