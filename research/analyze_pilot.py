"""Summarize the frozen pilot; no fitting, optimization, or new simulation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    paths = [ROOT / "certificate_pilot_v1.json", ROOT / "certificate_high_weight_v2.json"]
    datasets = [json.loads(path.read_text()) for path in paths]
    summaries = []
    for path, data in zip(paths, datasets):
        for profile in sorted({row["profile"] for row in data["cases"]}):
            for radius in (0., .01):
                rows = [row for row in data["cases"] if row["profile"] == profile and row["radius"] == radius]
                complete = [row for row in rows if row["status"] == "complete"]
                summaries.append({
                    "dataset": path.name, "profile": profile, "radius": radius,
                    "cases": len(rows), "complete": len(complete),
                    "energy_bound_at_most_0.01": sum(row["energy_bound_suffix"] <= .01 for row in complete),
                    "gradient_max_bound_at_most_0.05": sum(max(row["gradient_bound_suffix"]) <= .05 for row in complete),
                    "median_actual_center_energy_error": float(np.median([row["actual_energy_error"] for row in complete])),
                    "median_energy_bound": float(np.median([row["energy_bound_suffix"] for row in complete])),
                    "median_certificate_seconds": float(np.median([row["seconds"] for row in complete])),
                    "median_exact_seconds": float(np.median([row["exact_seconds_median"] for row in complete])),
                })
    output = {
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "threshold_status": "Illustrative retrospective thresholds, not preregistered success criteria.",
        "energy_convention": "H/|E| for unweighted H=sum ZZ; normalized MaxCut fraction error is half this energy error.",
        "timing_scope": "Certificate includes interval jets and bounds; exact reference also constructs QFI. These are implementation timings, not matched optimal algorithms.",
        "summary": summaries,
    }
    (ROOT / "pilot_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained")
    colors = {"optimized": "#b5423a", "small": "#277b70", "random": "#566ab2"}
    floor = 1e-12
    for profile, color in colors.items():
        rows = [row for row in datasets[0]["cases"] if row["profile"] == profile and row["radius"] == 0.]
        axes[0].scatter([max(floor, row["actual_energy_error"]) for row in rows],
                        [max(floor, row["energy_bound_suffix"]) for row in rows],
                        s=25, alpha=.7, color=color, label=profile)
    axes[0].plot([floor, 10], [floor, 10], color=".55", lw=1, ls=":")
    axes[0].axhline(.01, color=".3", lw=1, ls="--")
    axes[0].set(xscale="log", yscale="log", xlabel="Actual normalized energy error", ylabel="Point truncation bound",
                title="Low cutoffs: 54 point cases per profile", xlim=(floor / 2, 2), ylim=(floor / 2, 20))
    axes[0].legend(frameon=False, loc="lower right")
    for graph, color in [("petersen10", "#277b70"), ("cubic12", "#b5423a")]:
        for depth, marker in [(2, "o"), (3, "s")]:
            rows = sorted([row for data in datasets for row in data["cases"]
                           if row["workload"] == f"{graph}:p{depth}" and row["profile"] == "optimized" and row["radius"] == 0.], key=lambda row: row["cut"])
            axes[1].plot([row["cut"] for row in rows], [max(floor, row["energy_bound_suffix"]) for row in rows],
                         marker=marker, color=color, ls="-" if depth == 2 else "--", label=f"{graph}, p={depth}")
    axes[1].axhline(.01, color=".3", lw=1, ls=":")
    axes[1].set(yscale="log", xlabel="Retained Pauli weight cutoff", ylabel="Point truncation bound",
                title="Larger cutoffs help selectively", xticks=[2, 3, 4, 5, 6])
    axes[1].legend(frameon=False)
    fig.savefig(ROOT / "pilot_comparison.png", dpi=180)
    fig.savefig(ROOT / "pilot_comparison.pdf")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
