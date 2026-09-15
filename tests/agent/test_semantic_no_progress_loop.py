"""RED tests: repeated-test-run loops that the current stall guard misses.

Evidence (t_ec230c2a Run37, session 20260915_114653_8be3ba, codex-router-dev state.db):
396 identical `pytest tests/test_provider_status.py -q` terminal calls over 42 minutes
with NO source change in between (last patch 11:58:17, reruns 11:58:40-12:41:15).
The results differ only in pytest's per-run duration ("... in 2.55s"), so the
result-hash equality used by observe_call never holds and the guard stays silent
while the budget burns. These tests pin the contract that semantically-identical
failed test runs ARE caught (LLM-based equivalence, no magic keywords).
"""
import json

import pytest

import agent.semantic_no_progress as semantic_no_progress
from agent.tool_guardrails import (
    STALL_GUARD_IDENTICAL_CALL_THRESHOLD,
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
)


_ARGS = {
    "command": "cd /board && python3 -m pytest tests/test_provider_status.py -q 2>&1 | tail -2",
}
# Byte-differing, semantically identical failed run: only pytest's duration varies.
_RESULT_OUTPUT = (
    "FAILED tests/test_provider_status.py::UsageCreditsTest::"
    "test_usage_credits_rejects_invalid_shapes\n1 failed, 67 passed, 17 subtests "
    "passed in {seconds:.2f}s"
)


def _failed_run_result(i: int) -> str:
    return json.dumps({
        "output": _RESULT_OUTPUT.format(seconds=2.5 + (i % 7) * 0.01),
        "exit_code": 0, "error": None,
        "hint": "exit_code 0 here is the status of the last pipeline command",
    })


def _loop_controller(**overrides) -> ToolCallGuardrailController:
    config = ToolCallGuardrailConfig.from_mapping(
        {
            "warnings_enabled": True,
            "hard_stop_enabled": True,
            "warn_after": {"exact_failure": 2, "same_tool_failure": 3, "idempotent_no_progress": 2},
            "hard_stop_after": {"exact_failure": 5, "same_tool_failure": 8, "idempotent_no_progress": 5},
        },
        platform="gateway",
    )
    if overrides:
        config = ToolCallGuardrailConfig.from_mapping(
            {**{
                "warnings_enabled": True,
                "hard_stop_enabled": True,
                "warn_after": {"exact_failure": 2, "same_tool_failure": 3, "idempotent_no_progress": 2},
                "hard_stop_after": {"exact_failure": 5, "same_tool_failure": 8, "idempotent_no_progress": 5},
            }, **overrides},
            platform="gateway",
        )
    return ToolCallGuardrailController(config)


def test_byte_identical_results_halt_at_no_progress_threshold():
    """Control: the existing exact-equality path keeps working (period-1 streak)."""
    c = _loop_controller()
    for i in range(1, 397):
        r = json.dumps({**json.loads(_failed_run_result(0)), "output": "same"})
        c.after_call("terminal", _ARGS, r, failed=False)
        c.observe_call("terminal", _ARGS, r)
        if c.halt_decision is not None:
            assert c.halt_decision.code == "identical_call_streak_halt"
            assert i <= c.config.no_progress_block_after
            return
    pytest.fail("byte-identical replay loop was not halted")


def test_semantically_identical_failed_reruns_halt(monkeypatch):
    """RED: 396 reruns whose results differ only in run noise (duration) must halt.

    Without equivalence the whole Run37 budget (396 calls / 42 min) burned; the
    halt must land at the no-progress threshold, long before that. The judge is
    the production seam (agent.semantic_no_progress.ask_results_semantically_
    identical); it must be consulted exactly once — a fired halt is cached.
    """
    judge_results = []
    monkeypatch.setattr(
        semantic_no_progress, "ask_results_semantically_identical",
        lambda args_json, results: (judge_results.append(results), True)[1],
    )
    c = _loop_controller()
    halted_at = None
    for i in range(1, 397):
        r = _failed_run_result(i)
        decision = c.before_call("terminal", _ARGS)
        if decision.should_halt:
            halted_at = (i, decision.code)
            break
        c.after_call("terminal", _ARGS, r, failed=False)
        c.observe_call("terminal", _ARGS, r)
        if c.halt_decision is not None:
            halted_at = (i, c.halt_decision.code)
            break
    assert halted_at is not None, "semantically identical rerun loop ran to 396 calls unflagged"
    calls, code = halted_at
    assert code in {"identical_call_streak_halt", "identical_cycle_halt", "semantic_no_progress_halt"}
    assert calls <= c.config.no_progress_block_after * 2
    assert len(judge_results) == 1, "judge must be consulted once, then the halt is cached"


def test_legitimate_retest_after_source_change_is_not_halted():
    """GREEN guard for legit work: a real edit between failing runs resets the check —
    the fix->retest cycle stays allowed (regression test test_fix_retest_loop exists
    for the exact path; this mirrors it for the equivalence path)."""
    c = _loop_controller()
    for cycle in range(12):
        # failed run
        c.after_call("terminal", _ARGS, _failed_run_result(cycle * 2), failed=False)
        c.observe_call("terminal", _ARGS, _failed_run_result(cycle * 2))
        # source change lands (progress reset), then a DIFFERENT failing result
        c.after_call("patch", {"path": "x.py"}, json.dumps({"success": True}), failed=False)
        c.after_call("terminal", _ARGS, _failed_run_result(cycle * 2 + 1), failed=False)
        c.observe_call("terminal", _ARGS, _failed_run_result(cycle * 2 + 1))
        assert c.halt_decision is None, f"fix->retest cycle {cycle} was wrongly halted"


def test_varying_poll_results_are_never_halted():
    """A poller whose output genuinely changes (progress percentages) never halts."""
    c = _loop_controller()
    for i in range(1, 60):
        r = json.dumps({"status": "running", "progress": i})
        c.after_call("terminal", {"command": "poll-build"}, r, failed=False)
        c.observe_call("terminal", {"command": "poll-build"}, r)
        assert c.halt_decision is None, f"varying poll halted at call {i}"
