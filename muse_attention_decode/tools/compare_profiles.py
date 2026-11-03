"""Compare completed OKC profiles using paired ratios and one canonical baseline.

First argument is the current scored profile, followed by candidates. This is an
analysis-only estimate; validate any combined dispatch with a fresh OKC check.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("profiles", nargs="+", type=Path)
parser.add_argument("--min-gain", type=float, default=1.03)
args = parser.parse_args()


def shape(case):
    return (case["batch"], case.get("q_len", 1), case["heads"], case["layer"],
            case.get("page", 64), case["ctx"])


profiles = []
for path in args.profiles:
    data = json.loads(path.read_text())
    if not data["summary"].get("passed"):
        raise ValueError(f"Not a passing profile: {path}")
    name = Path(data["submission"]).stem
    profiles.append((name, {shape(b["case"]): b for b in data["benchmarks"]}))

current_name, canonical = profiles[0]
loads = defaultdict(lambda: [0., 0., 0.])
print(" B  L layer       current_us best_us    gain selected")
for key, current in canonical.items():
    baseline_ns = current["baseline_ns"]
    old_ns = baseline_ns / current["paired_ratio"]
    choices = [(baseline_ns / cases[key]["paired_ratio"], name)
               for name, cases in profiles if key in cases]
    new_ns, name = min(choices)
    gain = old_ns / new_ns
    if gain < args.min_gain:
        new_ns, name = old_ns, current_name
    case = current["case"]
    b, length, _, layer, _, _ = key
    print(f"{b:2} {length:2} {layer:9} {old_ns/1000:10.2f} {new_ns/1000:7.2f} {old_ns/new_ns:7.3f} {name}")
    totals = loads[case["load"]]
    weight = case["weight"]
    totals[0] += baseline_ns * weight
    totals[1] += old_ns * weight
    totals[2] += new_ns * weight


def score(index):
    return math.exp(sum(math.log(v[0] / v[index]) for v in loads.values()) / len(loads))


print(f"Current normalized speedup: {score(1):.5f}x")
print(f"Estimated shape-wise combination: {score(2):.5f}x")
print("Estimate only; do not promote an unmeasured combination.")
