import json
import argparse
from pathlib import Path


def read_trajectory(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def get_usage(response):
    """Extract provider-reported token usage from one API response."""
    if not isinstance(response, dict):
        return {}

    usage = response.get("usage") or {}

    return {
        "prompt": usage.get("prompt_tokens", 0),
        "completion": usage.get("completion_tokens", 0),
        "total": usage.get(
            "total_tokens",
            usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
        ),
    }


def aggregate_usage(responses):
    totals = {"prompt": 0, "completion": 0, "total": 0}

    for response in responses:
        usage = get_usage(response)

        for key in totals:
            totals[key] += usage[key]

    return totals


def analyze_trajectory(path):
    data = read_trajectory(path)

    agent_usage = aggregate_usage(data.get("responses", []))

    compaction_responses = [
        event.get("compaction_response", {}) for event in data.get("compactions", [])
    ]

    compaction_usage = aggregate_usage(compaction_responses)

    combined = {key: agent_usage[key] + compaction_usage[key] for key in agent_usage}

    events = data.get("compactions", [])

    return {
        "name": Path(path).name,
        "steps": len(data.get("responses", [])),
        "compaction_events": len(events),
        "agent_usage": agent_usage,
        "compaction_usage": compaction_usage,
        "combined": combined,
        "estimated_before": sum(
            event.get("estimated_tokens_before", 0) for event in events
        ),
        "estimated_after": sum(
            event.get("estimated_tokens_after", 0) for event in events
        ),
    }


def format_report(baseline, compacted, baseline_result, compacted_result):
    b = baseline["combined"]
    c = compacted["combined"]

    saved = b["total"] - c["total"]
    reduction = saved / b["total"] * 100 if b["total"] else 0

    context_before = compacted["estimated_before"]
    context_after = compacted["estimated_after"]

    context_reduction = (
        (context_before - context_after) / context_before * 100 if context_before else 0
    )

    rows = [
        ("Agent steps", baseline["steps"], compacted["steps"]),
        (
            "Compaction events",
            baseline["compaction_events"],
            compacted["compaction_events"],
        ),
        (
            "Agent prompt tokens",
            baseline["agent_usage"]["prompt"],
            compacted["agent_usage"]["prompt"],
        ),
        (
            "Compaction prompt tokens",
            baseline["compaction_usage"]["prompt"],
            compacted["compaction_usage"]["prompt"],
        ),
        ("Total prompt tokens", b["prompt"], c["prompt"]),
        ("Total completion tokens", b["completion"], c["completion"]),
        ("Total tokens", b["total"], c["total"]),
        ("SWE-bench result", baseline_result, compacted_result),
    ]

    lines = [
        "# Token Usage Analysis",
        "",
        "## Experiment",
        "",
        f"- Baseline trajectory: `{baseline['name']}`",
        f"- Compacted trajectory: `{compacted['name']}`",
        "",
        "## Results",
        "",
        "| Metric | Baseline | Compacted |",
        "|---|---:|---:|",
    ]

    for label, baseline_value, compacted_value in rows:
        lines.append(
            f"| {label} | {baseline_value:,} | {compacted_value:,} |"
            if isinstance(baseline_value, (int, float))
            and isinstance(compacted_value, (int, float))
            else f"| {label} | {baseline_value} | {compacted_value} |"
        )

    lines.extend(
        [
            "",
            "## Token Savings",
            "",
            f"- Baseline total: **{b['total']:,} tokens**",
            f"- Compacted total: **{c['total']:,} tokens**",
            f"- Tokens saved: **{saved:,}**",
            f"- Total token reduction: **{reduction:.2f}%**",
            "",
            "## Compaction Analysis",
            "",
            f"- Compaction events: {compacted['compaction_events']}",
            f"- Estimated tokens before compaction (summed snapshots): "
            f"{context_before:,}",
            f"- Estimated tokens after compaction (summed snapshots): "
            f"{context_after:,}",
            f"- Snapshot reduction: **{context_reduction:.2f}%**",
            "",
            "## Observations",
            "",
            f"1. The baseline used {baseline['steps']} agent steps.",
            f"2. The compacted run used {compacted['steps']} agent steps.",
            f"3. Context compaction occurred {compacted['compaction_events']} times.",
            f"4. Compaction calls consumed "
            f"{compacted['compaction_usage']['total']:,} additional tokens.",
            f"5. Total token usage decreased by {reduction:.2f}%.",
            "",
            "## Conclusion",
            "",
            "This experiment compares context compaction against a "
            "full-context baseline using the same SWE-bench instance. "
            "Token totals include both normal agent calls and compaction "
            "API calls. The recorded context estimates are summed across "
            "compaction snapshots and are not unique-token savings.",
            "",
            "SWE-bench evaluation results are supplied separately because "
            "they are not stored in the trajectory JSON.",
            "",
        ]
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate token usage analysis from agent trajectories."
    )

    parser.add_argument(
        "--baseline",
        default="artifacts/django__django-15368-baseline-trajectory.json",
    )
    parser.add_argument(
        "--compacted",
        default="artifacts/django__django-15368-trajectory.json",
    )
    parser.add_argument(
        "--output",
        default="artifacts/token-usage-analysis.md",
    )
    parser.add_argument(
        "--baseline-result",
        default="Not supplied",
    )
    parser.add_argument(
        "--compacted-result",
        default="Not supplied",
    )

    args = parser.parse_args()

    baseline = analyze_trajectory(args.baseline)
    compacted = analyze_trajectory(args.compacted)

    report = format_report(
        baseline,
        compacted,
        args.baseline_result,
        args.compacted_result,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")

    print(f"Analysis generated: {output}")
    print(f"Baseline tokens: {baseline['combined']['total']:,}")
    print(f"Compacted tokens: {compacted['combined']['total']:,}")
    print(f"Compaction events: {compacted['compaction_events']}")


if __name__ == "__main__":
    main()
