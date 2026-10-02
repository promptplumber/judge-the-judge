"""The LLM relevance judge: one (query, product) pair in, one structured verdict out.

Every outcome is one of three statuses, and they are never mixed:
  ok             - a valid verdict (is_relevant is True/False)
  parse_failure  - the model answered but not in the required shape (no verdict is recorded)
  infra_failure  - timeout / connection / rate-limit / 5xx (the model never really answered)
Only `ok` results count as judgments; the other two are tracked separately downstream.

The request builder and the response parser are separate from the network call so the
Batch API path can reuse them unchanged.
"""
import json
from dataclasses import dataclass

import openai
from dotenv import load_dotenv
from openai import OpenAI

from src.config import ROOT, load_config
from src.costs import log_api_call

OK, PARSE_FAILURE, INFRA_FAILURE = "ok", "parse_failure", "infra_failure"

SYSTEM_PROMPTS = {
    "v1": (
        "You decide whether a product is a relevant result for a shopping query.\n"
        "You will be given a shopping query and a product title. Answer one yes/no rubric "
        "question:\n"
        "  is_relevant: Would a shopper searching for the query consider this product a "
        "directly relevant result they might buy?\n"
        "Judge the query exactly as written, including any constraints it states (exclusions "
        "such as 'without X' or 'not Y', sizes, quantities, colors, brands). Use only the "
        "information in the product title.\n"
        "Respond with a JSON object with exactly these fields:\n"
        '  "is_relevant": true or false,\n'
        '  "justification": one sentence explaining the verdict,\n'
        '  "evidence": the specific product attribute (quoted or paraphrased from the title) '
        "that drove the verdict."
    ),
}

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "relevance_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "is_relevant": {"type": "boolean"},
                "justification": {"type": "string"},
                "evidence": {"type": "string"},
            },
            "required": ["is_relevant", "justification", "evidence"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class JudgeResult:
    status: str
    is_relevant: bool | None = None   # set only when status == "ok"
    justification: str | None = None
    evidence: str | None = None
    raw: str | None = None            # raw model text, kept for debugging parse failures
    error: str | None = None          # why it was a parse/infra failure


def build_request_body(query: str, product_title: str, cfg: dict | None = None) -> dict:
    """Chat Completions request body. Also used verbatim as the Batch API request body."""
    j = (cfg or load_config())["judge"]
    return {
        "model": j["model"],
        "temperature": j["temperature"],
        "max_completion_tokens": j["max_completion_tokens"],
        "response_format": RESPONSE_SCHEMA,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[j["prompt_version"]]},
            {"role": "user", "content": f"Query: {query}\nProduct title: {product_title}"},
        ],
    }


def parse_judgment(content: str | None, finish_reason: str | None = "stop") -> JudgeResult:
    """Parse the model's text into a JudgeResult. Anything malformed is a parse_failure,
    never a verdict (so it can't be mistaken for the judge being wrong)."""
    def fail(why: str) -> JudgeResult:
        return JudgeResult(PARSE_FAILURE, raw=content, error=why)

    if finish_reason == "length":
        return fail("truncated (finish_reason=length)")
    if not content:
        return fail("empty response")
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as e:
        return fail(f"invalid JSON: {e}")
    if not isinstance(obj, dict):
        return fail("JSON is not an object")
    if not isinstance(obj.get("is_relevant"), bool):  # "yes"/1/null are not verdicts
        return fail("is_relevant missing or not a boolean")
    for key in ("justification", "evidence"):
        if not isinstance(obj.get(key), str):
            return fail(f"{key} missing or not a string")
    return JudgeResult(OK, obj["is_relevant"], obj["justification"], obj["evidence"], raw=content)


def parse_completion(completion: dict) -> JudgeResult:
    """Parse a Chat Completions response in dict form (SDK .model_dump() or a Batch API
    response body)."""
    try:
        choice = completion["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError):
        return JudgeResult(PARSE_FAILURE, error="no choices in response")
    if message.get("refusal"):
        return JudgeResult(PARSE_FAILURE, raw=message["refusal"], error="model refused")
    return parse_judgment(message.get("content"), choice.get("finish_reason"))


def make_client(cfg: dict | None = None) -> OpenAI:
    load_dotenv(ROOT / ".env")
    j = (cfg or load_config())["judge"]
    return OpenAI(timeout=j["timeout_s"], max_retries=j["max_retries"])


# Transient problems on our side of the fence. Anything else (e.g. a 400 from a bad request)
# is a bug and should raise loudly rather than be silently filed as an infra failure.
_INFRA_ERRORS = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)


def judge_pair(query: str, product_title: str, cfg: dict | None = None,
               client: OpenAI | None = None, tag: str = "adhoc") -> JudgeResult:
    """`tag` labels this call in the cost log (the harness passes the run_id)."""
    cfg = cfg or load_config()
    client = client or make_client(cfg)
    try:
        completion = client.chat.completions.create(**build_request_body(query, product_title, cfg))
    except _INFRA_ERRORS as e:
        return JudgeResult(INFRA_FAILURE, error=f"{type(e).__name__}: {e}")
    data = completion.model_dump()
    result = parse_completion(data)
    # Parse failures still consumed tokens, so every answered call is logged.
    log_api_call(data.get("usage") or {}, cfg["judge"]["model"], tag, status=result.status, cfg=cfg)
    return result


# --- Checkpoint 2 -------------------------------------------------------------------------

HAND_PICKED = [  # (query, product title, expected is_relevant)
    ("wireless bluetooth headphones",
     "Sony WH-1000XM4 Wireless Noise Canceling Over-Ear Bluetooth Headphones, Black", True),
    ("men's running shoes size 10",
     "Nike Men's Air Zoom Pegasus 40 Running Shoes, Size 10", True),
    ("usb c charging cable",
     "Anker USB C to USB C Cable (6 ft, 60W), Fast Charging for MacBook, iPad, Galaxy", True),
    ("stainless steel water bottle",
     "Funko Pop! Star Wars Darth Vader Vinyl Bobblehead Figure", False),
    ("gaming laptop",
     "Bamboo Cutting Board with Juice Groove, Extra Large Kitchen Chopping Board", False),
]

MALFORMED = [  # (label, content, finish_reason)
    ("not JSON", "Sure! I think this product is relevant.", "stop"),
    ("JSON but wrong shape", '["is_relevant", true]', "stop"),
    ("missing field", '{"is_relevant": true, "justification": "matches"}', "stop"),
    ("verdict is a string", '{"is_relevant": "yes", "justification": "a", "evidence": "b"}', "stop"),
    ("truncated", '{"is_relevant": tr', "length"),
    ("empty", "", "stop"),
]


if __name__ == "__main__":
    cfg = load_config()
    client = make_client(cfg)
    print(f"model={cfg['judge']['model']} prompt_version={cfg['judge']['prompt_version']} "
          f"temperature={cfg['judge']['temperature']}\n")

    print("== Hand-picked pairs ==")
    correct = 0
    for query, title, expected in HAND_PICKED:
        r = judge_pair(query, title, cfg, client)
        mark = "OK " if (r.status == OK and r.is_relevant == expected) else "BAD"
        correct += mark == "OK "
        print(f"[{mark}] expected={expected}  status={r.status}  verdict={r.is_relevant}")
        print(f"      query:         {query}\n      product:       {title}")
        print(f"      justification: {r.justification}\n      evidence:      {r.evidence}")
        if r.error:
            print(f"      error:         {r.error}")
    print(f"\n{correct}/{len(HAND_PICKED)} hand-picked pairs match expectation\n")

    print("== Malformed responses (must be parse_failure, never a verdict) ==")
    all_caught = True
    for label, content, finish in MALFORMED:
        r = parse_judgment(content, finish)
        caught = r.status == PARSE_FAILURE and r.is_relevant is None
        all_caught &= caught
        print(f"[{'OK ' if caught else 'BAD'}] {label:22s} -> status={r.status} is_relevant={r.is_relevant} ({r.error})")
    print("\nall malformed responses caught as parse failures:", all_caught)
