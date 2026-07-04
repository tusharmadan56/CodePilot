"""Smoke test for model connectivity. Run from repo root: python -m scripts.smoke"""

from dotenv import load_dotenv

from agent.llm import build_llm, DEFAULT_MODEL


def main() -> None:
    load_dotenv()
    llm = build_llm()
    resp = llm.invoke("Reply with exactly this line and nothing else: CodePilot online.")
    print(f"[{DEFAULT_MODEL}] {resp.content}")


if __name__ == "__main__":
    main()
