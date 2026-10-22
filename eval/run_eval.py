"""
Golden-set evaluation for the triage stage.

Usage (from the repo root):
    python -m eval.run_eval              # rules + SLM if Ollama is running
    python -m eval.run_eval --rules-only # deterministic baseline, no model

The labels are SYNTHETIC and written by the team. They are not clinician-reviewed;
report results as such. gold_min_urgency is the lowest acceptable urgency:
scoring below it is UNDER-triage, scoring above it is OVER-triage.
"""
import argparse, json, pathlib
from collections import defaultdict
from backend import config
from backend.pipeline import triage, flagging

RANK = {l: i for i, l in enumerate(config.URGENCY_LEVELS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules-only", action="store_true")
    args = ap.parse_args()
    if args.rules_only:
        config.SLM_PROVIDER = "none"
        triage.SLM_PROVIDER = "none"

    cases = json.loads((pathlib.Path(__file__).parent / "golden_cases.json").read_text())
    by_lang = defaultdict(lambda: {"n": 0, "under": 0, "over": 0, "crit": 0, "crit_hit": 0})
    misses = []
    for c in cases:
        flags = flagging.run_all_flags([], [], c["text"])
        r = triage.score_case(c["id"], c["text"], flags)
        got, gold = RANK[r.urgency_level], RANK[c["gold_min_urgency"]]
        for key in (c["language"], "ALL"):
            s = by_lang[key]
            s["n"] += 1
            s["under"] += got < gold
            s["over"] += got > gold
            if c["gold_min_urgency"] == "critical":
                s["crit"] += 1
                s["crit_hit"] += got == RANK["critical"]
        if got < gold:
            misses.append((c["id"], c["language"], c["gold_min_urgency"], r.urgency_level, c["text"]))

    mode = "rules-only" if args.rules_only else "rules + SLM (falls back to manual review if SLM is down)"
    print(f"Mode: {mode}\n{'lang':<7}{'n':>4}{'crit recall':>14}{'under':>8}{'over':>7}")
    for lang in ["ALL", "en", "hi", "mixed"]:
        s = by_lang[lang]
        rec = f"{s['crit_hit']}/{s['crit']}" if s["crit"] else "-"
        print(f"{lang:<7}{s['n']:>4}{rec:>14}{s['under']:>8}{s['over']:>7}")
    print("\nUnder-triaged cases (safety misses):")
    for m in misses:
        print("  ", m)
    if not misses:
        print("   none")


if __name__ == "__main__":
    main()
