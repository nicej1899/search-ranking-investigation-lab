"""Optional live LLM tool-calling driver over the same Lab tools.

Enabled only when ANTHROPIC_API_KEY is already configured in the environment (never asked for during a
presentation). NOT exercised in this build: no provider was configured where it was developed, so every
trace shipped with the demo comes from the deterministic scripted workflow (agent.py). The tool outputs,
provenance envelope and export are identical; only the step-selection policy differs.

Usage: ANTHROPIC_API_KEY=... python -m srlab.agent_llm [case_id]
"""
from __future__ import annotations

import json
import os
import sys

from .agent import Lab
from .common import CASES, load_json

TOOLS = [
    {"name": "load_case", "description": "Load member history, query and candidates.", "input_schema": {"type": "object", "properties": {"case_id": {"type": "string"}}, "required": ["case_id"]}},
    {"name": "score_candidates", "description": "Score candidates with a ranker (F0/F1/Fctrl), optionally after a recorded intervention.",
     "input_schema": {"type": "object", "properties": {"case_id": {"type": "string"}, "model_id": {"type": "string"}, "intervention_id": {"type": "string"}}, "required": ["case_id"]}},
    {"name": "explain_history", "description": "TemGX-adapted explanation of A ranked above B.",
     "input_schema": {"type": "object", "properties": {"case_id": {"type": "string"}, "A_index": {"type": "integer"}, "B_index": {"type": "integer"}, "budget": {"type": "integer"}}, "required": ["case_id", "A_index", "B_index"]}},
    {"name": "replay_intervention", "description": "Remove member events and re-run the frozen model.",
     "input_schema": {"type": "object", "properties": {"case_id": {"type": "string"}, "event_ids": {"type": "array", "items": {"type": "integer"}}, "A_index": {"type": "integer"}, "B_index": {"type": "integer"}}, "required": ["case_id", "event_ids"]}},
    {"name": "analyze_motifs", "description": "CATGX-adapted motif scores for the case plus DEV comparable-context aggregate.",
     "input_schema": {"type": "object", "properties": {"case_id": {"type": "string"}, "A_index": {"type": "integer"}, "B_index": {"type": "integer"}}, "required": ["case_id", "A_index", "B_index"]}},
    {"name": "propose_variant", "description": "Pick ONE allowed change: history_gate | global_history_scale | bounded_popularity | content_feature.",
     "input_schema": {"type": "object", "properties": {"diagnosis_id": {"type": "string"}, "allowed_change": {"type": "string"}}, "required": ["diagnosis_id", "allowed_change"]}},
    {"name": "train_variant", "description": "Fit a proposed variant on DEV. Pass the outputs of propose_variant.",
     "input_schema": {"type": "object", "properties": {"variant_config": {"type": "object"}}, "required": ["variant_config"]}},
    {"name": "evaluate_variant", "description": "Evaluate on qi-dev-v1 or qi-final-v1 (final is scored once, then cached).",
     "input_schema": {"type": "object", "properties": {"model_id": {"type": "string"}, "evaluation_protocol_id": {"type": "string"}}, "required": ["model_id", "evaluation_protocol_id"]}},
    {"name": "compare_variants", "description": "Paired member-bootstrap comparison by slice.",
     "input_schema": {"type": "object", "properties": {"baseline_id": {"type": "string"}, "variant_id": {"type": "string"}, "evaluation_id": {"type": "string"}}, "required": ["baseline_id", "variant_id", "evaluation_id"]}},
]
SYSTEM = ("You are investigating a search-ranking failure with real tools. Cite artifact_ids for every claim, separate evidence from "
          "hypothesis, change one mechanism at a time, select on DEV before scoring FINAL once, and report regressions or no-improvement honestly.")


def _compact(o, depth=0):
    s = json.dumps(o, default=str)
    return s if len(s) < 6000 else s[:6000] + "...<truncated>"


def run(case_id: str | None = None, max_turns: int = 24):
    import anthropic  # imported lazily; optional dependency
    sel = load_json(CASES / "selected_cases.json")["primary"]
    case_id = case_id or sel["case_id"]
    lab = Lab("inv-llm", "live LLM tool-calling agent")
    client = anthropic.Anthropic()
    msgs = [{"role": "user", "content": f"Investigate case {case_id}: A_index={sel['A_index']}, B_index={sel['B_index']}. "
                                        "Diagnose why A outranks B, test one model change, and recommend the next experiment."}]
    final_text = ""
    for _ in range(max_turns):
        r = client.messages.create(model=os.environ.get("LAB_LLM_MODEL", "claude-sonnet-5-5"), max_tokens=2000, system=SYSTEM, tools=TOOLS, messages=msgs)
        msgs.append({"role": "assistant", "content": r.content})
        uses = [b for b in r.content if b.type == "tool_use"]
        final_text = "".join(b.text for b in r.content if b.type == "text")
        if not uses:
            break
        results = []
        for u in uses:
            rec = getattr(lab, u.name)(**u.input)
            results.append({"type": "tool_result", "tool_use_id": u.id, "content": _compact({k: rec[k] for k in ("artifact_id", "outputs", "limitations", "errors", "cache")})})
        msgs.append({"role": "user", "content": results})
    return lab.export_investigation(extra={"case_id": case_id, "A_index": sel["A_index"], "B_index": sel["B_index"], "llm_final_text": final_text,
                                           "narrative": [{"kind": "llm", "text": final_text, "cites": []}],
                                           "diagnosis": {"evidence": [], "hypotheses": [], "caveats": ["LLM narrative; verify against cited artifacts."]}})


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("No provider configured (ANTHROPIC_API_KEY unset). Use the scripted workflow: python -m srlab.agent")
    print(run(sys.argv[1] if len(sys.argv) > 1 else None))
