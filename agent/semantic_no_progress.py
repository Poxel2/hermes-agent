"""LLM-based semantic no-progress evaluation for the stall guard.

The byte-exact detectors in ``agent.tool_guardrails`` miss the most expensive loop shape:
a failed test/CI command re-run N times with NO source change between runs, whose results
differ only in per-run noise (pytest's ``in 2.55s`` timing, a timestamp, a percentage).
The Run37 incident (session 20260915_114653_8be3ba) burned 396 identical pytest calls /
42 minutes of budget exactly this way — every hash-based detector stayed silent because
no two results were byte-equal.

This module answers ONE question with a cheap auxiliary-LLM call: "are these recent
same-call results semantically identical?" The mechanical hashes stay the evidence
collector; semantics is judged by an LLM, never by keyword heuristics (contract: no
magic-keyword matching). Fail-open everywhere: an unavailable/broken auxiliary provider
must never halt legitimate work.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Number of same-call results handed to the judge. Recent tail only: the verdict is
# "is the model re-running the same failing thing", which the last few runs answer.
EVIDENCE_RESULT_COUNT = 3
# Results above this length are head+tail sampled; the loop signature (same failure,
# same counts, only noise differing) is always at the edges.
_MAX_RESULT_CHARS = 1200
# One cheap judge call per streak formation, then a cached verdict; a re-judge only
# happens when the streak would grow past the next threshold.
_JUDGE_TASK = "stall_guard_semantic"
_JUDGE_TIMEOUT_SECONDS = 45
# After the auxiliary route fails with "no provider configured", stop trying for the
# process lifetime: the auto-detection chain costs ~0.4s per attempt in a no-credential
# environment, and a guard that probes a dead provider on every tool call is its own
# stall. Re-checked never (config changes need a restart anyway).
_no_provider: bool = False

_VERDICT_TRUE = "true"
_VERDICT_FALSE = "false"

_JUDGE_PROMPT = (
    "You are judging whether an AI agent is stuck in an unproductive loop.\n"
    "The agent made the same tool call N times in a row (identical arguments). "
    "Below are the results of the most recent calls.\n\n"
    "Decide: are these results SEMANTICALLY IDENTICAL — i.e. they report the same "
    "state/outcome, differing only in per-run noise such as durations, timestamps, "
    "percentages, ids, or ordering?\n"
    '- "1 failed, 67 passed ... in 2.55s" and "1 failed, 67 passed ... in 2.58s" '
    "are semantically identical (same test outcome).\n"
    "- Results reporting DIFFERENT outcomes (different failure, different counts, "
    "new/changed content, genuine progress) are NOT semantically identical.\n\n"
    "Reply with exactly one JSON object and nothing else:\n"
    '{"semantically_identical": true|false, "reason": "<max 20 words>"}'
)


def _sample_result(result: str) -> str:
    """Head+tail sample of a long result; the failure summary lives at the edges."""
    if len(result) <= _MAX_RESULT_CHARS:
        return result
    head = result[: _MAX_RESULT_CHARS // 2]
    tail = result[-_MAX_RESULT_CHARS // 2:]
    return f"{head}\n[...{len(result) - _MAX_RESULT_CHARS} chars omitted...]\n{tail}"


def build_judge_messages(args_json: str, results: List[str]) -> List[dict]:
    """Build the auxiliary-judge request. Pure: exported for tests and reuse."""
    evidence = "\n\n".join(
        f"--- result {i + 1} of {len(results)} ---\n{_sample_result(r)}"
        for i, r in enumerate(results)
    )
    return [
        {"role": "system", "content": _JUDGE_PROMPT},
        {
            "role": "user",
            "content": (
                f"tool arguments (identical on every call):\n{args_json}\n\nresults:\n{evidence}"
            ),
        },
    ]


def parse_judge_verdict(content: Any) -> Optional[bool]:
    """Parse the judge reply; None = unparsable (caller treats as no-verdict, fail open)."""
    if not isinstance(content, str) or not content.strip():
        return None
    text = content.strip()
    # Tolerate a fenced or prose-wrapped JSON object; the schema asked for bare JSON.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    verdict = data.get("semantically_identical")
    if isinstance(verdict, bool):
        return verdict
    if isinstance(verdict, str):
        lowered = verdict.strip().lower()
        if lowered == _VERDICT_TRUE:
            return True
        if lowered == _VERDICT_FALSE:
            return False
    return None


def ask_results_semantically_identical(args_json: str, results: List[str]) -> Optional[bool]:
    """One auxiliary-LLM judge call. True/False = verdict, None = judge unavailable or
    unparsable (fail open — never halt on a broken judge)."""
    global _no_provider
    if not results or _no_provider:
        return None
    try:
        from agent.auxiliary_client import call_llm

        response = call_llm(
            task=_JUDGE_TASK,
            messages=build_judge_messages(args_json, results),
            # Thinking-only deployments (Fireworks GLM-5.x) cannot disable thinking and
            # spend 200-500 tokens before the verdict; too small a cap truncates the JSON
            # payload into finish_reason="length" (live-probed: 64 fails, 256 lands).
            max_tokens=512,
            temperature=0.0,
            timeout=_JUDGE_TIMEOUT_SECONDS,
            # reasoning_config deliberately NOT set: a disable is unexpressible on strict
            # schemas (Fireworks 400s on the field) and thinking-only models ignore it.
        )
    except Exception as exc:
        # "No LLM provider configured" is terminal for the process: the auto-detection
        # chain costs ~0.4s per attempt in a no-credential environment, so cache the
        # negative verdict instead of re-probing on every tool call.
        if "No LLM provider configured" in str(exc):
            _no_provider = True
            logger.info("Semantic no-progress judge disabled: no auxiliary provider configured.")
        else:
            logger.debug("semantic no-progress judge call failed: %s", exc)
        return None
    try:
        content = response.choices[0].message.content
    except Exception:
        logger.debug("semantic no-progress judge returned no content", exc_info=True)
        return None
    return parse_judge_verdict(content)
