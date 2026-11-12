"""저장된 Item-Based 평가 결과로 그래프·비교표·실행 결과 요약을 생성한다."""

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReportData:
    k_results: list[dict]
    gamma_results: list[dict]
    baseline: dict
    adjusted: dict
    sources: list[Path]


def _read(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"결과 파일을 읽을 수 없습니다: {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"JSON 객체가 필요합니다: {path}")
    return result


def _validate(result: dict, split: str, top_k: int, gamma: float, source: Path) -> None:
    expected = {"split": split, "top_k": top_k, "gamma": gamma, "top_n": 10}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"{source}: {key}는 {value}이어야 합니다.")
    for key in ("algorithm", "weighting"):
        if not isinstance(result.get(key), str) or not result[key]:
            raise ValueError(f"{source}: {key} 정보가 필요합니다.")
    for key in ("num_users", "target_count", "recommendation_count"):
        value = result.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{source}: 유효한 {key}가 필요합니다.")
    if result["recommendation_count"] != 10 * result["num_users"]:
        raise ValueError(f"{source}: 사용자별 추천 10곡 조건과 다릅니다.")
    for key in ("recall_at_10", "arp"):
        value = result.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{source}: 유효하지 않은 지표 {key}")
    if result["recall_at_10"] > 1:
        raise ValueError(f"{source}: Recall@10은 1 이하여야 합니다.")


def load_report_data(
    results_dir: Path, k_values=(20, 50, 100), gammas=(0.0, 0.1, 0.25), selected_k=20,
) -> ReportData:
    results_dir = Path(results_dir)
    if not k_values or not gammas or any(k < 1 for k in k_values):
        raise ValueError("양수 K와 gamma 목록이 필요합니다.")
    if any(not 0 <= gamma <= 1 for gamma in gammas):
        raise ValueError("gamma는 0~1이어야 합니다.")
    if len(set(k_values)) != len(k_values) or len(set(gammas)) != len(gammas):
        raise ValueError("K·gamma 목록에 중복이 있습니다.")
    comparison_path = results_dir / f"test_k{selected_k}" / "comparison.json"
    comparison = _read(comparison_path)
    if comparison.get("split") != "test" or comparison.get("top_k") != selected_k:
        raise ValueError(f"{comparison_path}: 테스트 설정이 다릅니다.")
    baseline, adjusted = comparison.get("baseline"), comparison.get("adjusted")
    adjusted_gamma = comparison.get("adjusted_gamma")
    if not isinstance(baseline, dict) or not isinstance(adjusted, dict):
        raise ValueError(f"{comparison_path}: 보정 전후 결과가 필요합니다.")
    if type(adjusted_gamma) not in (int, float) or not 0 < adjusted_gamma <= 1:
        raise ValueError(f"{comparison_path}: 유효한 보정 gamma가 필요합니다.")
    _validate(baseline, "test", selected_k, 0.0, comparison_path)
    _validate(adjusted, "test", selected_k, adjusted_gamma, comparison_path)
    if selected_k not in k_values or adjusted_gamma not in gammas:
        raise ValueError("선택한 K·gamma가 검증 비교 목록에 포함되어야 합니다.")
    shared_keys = ("algorithm", "weighting", "num_users", "target_count", "candidate_rule")
    if any(baseline.get(key) != adjusted.get(key) for key in shared_keys):
        raise ValueError(f"{comparison_path}: 보정 전후 평가 조건이 다릅니다.")
    sources = [comparison_path]

    def validation(k, gamma):
        gamma_label = f"{gamma:g}".replace(".", "")
        path = results_dir / f"validation_k{k}_gamma{gamma_label}" / "summary.json"
        result = _read(path)
        _validate(result, "validation", k, gamma, path)
        if any(result[key] != baseline[key] for key in ("algorithm", "weighting")):
            raise ValueError(f"{path}: 테스트와 알고리즘·입력 변환이 다릅니다.")
        sources.append(path)
        return result

    k_results = [validation(k, 0.0) for k in k_values]
    gamma_results = [validation(selected_k, gamma) for gamma in gammas]
    reference = k_results[0]
    for result in k_results + gamma_results:
        if any(result.get(key) != reference.get(key)
               for key in ("num_users", "target_count", "candidate_rule")):
            raise ValueError("검증 결과의 평가 사용자·정답·후보 조건이 일치하지 않습니다.")
    return ReportData(k_results, gamma_results, baseline, adjusted, list(dict.fromkeys(sources)))


def format_execution_results(data: ReportData) -> str:
    lines = ["TuneFit Item-Based CF — Code execution results",
             "Source: saved evaluation JSON files; no training or evaluation rerun.",
             f"Algorithm: {data.baseline['algorithm']}",
             f"Weighting: {data.baseline['weighting']}", ""]
    for title, results in (("Validation: K comparison (gamma=0)", data.k_results),
                           ("Validation: gamma comparison", data.gamma_results),
                           ("Test: before / after adjustment", [data.baseline, data.adjusted])):
        lines.append(title)
        for result in results:
            lines.append(f"K={result['top_k']}, gamma={result['gamma']:g}, "
                         f"users={result['num_users']:,}: "
                         f"Recall@10={result['recall_at_10']:.6f}, ARP={result['arp']:.6f}")
        lines.append("")
    before, after = data.baseline, data.adjusted
    if before["recall_at_10"]:
        retention = after["recall_at_10"] / before["recall_at_10"]
        lines.append(f"Test Recall retention: {retention * 100:.2f}%")
        lines.append(f"Test Recall change: {(retention - 1) * 100:+.2f}%")
    else:
        lines.append("Test Recall retention: N/A (baseline Recall is zero)")
    if before["arp"]:
        reduction = 1 - after["arp"] / before["arp"]
        lines.append(f"Test ARP reduction: {reduction * 100:.2f}%")
    else:
        lines.append("Test ARP reduction: N/A (baseline ARP is zero)")
    lines += ["", "Popularity means unique listeners in the training sample.",
              "Lower ARP alone does not establish improved user satisfaction.",
              "", "Input files:"]
    lines.extend(str(path) for path in data.sources)
    return "\n".join(lines) + "\n"


def _plot_pair(results, labels, title, note, highlights, output: Path) -> None:
    # GUI를 열지 않고 이미지 파일만 생성한다.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tunefit-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    colors = ["#19364F" if highlighted else "#93A7B8" for highlighted in highlights]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        try:
            fig.suptitle(title, fontsize=17, fontweight="bold", color="#19364F", y=.97)
            for ax, key, label, precision in zip(
                axes, ("recall_at_10", "arp"),
                ("Recall@10 (higher is better)", "ARP (lower is better)"), (6, 3),
            ):
                values = [result[key] for result in results]
                bars = ax.bar(range(len(labels)), values, color=colors, width=.55, zorder=3)
                ax.set_xticks(range(len(labels)), labels)
                ax.set_title(label, fontsize=12, pad=14)
                ax.set_ylim(0, max(values) * 1.3 if max(values) else 1)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                ax.grid(axis="y", color="#E4E9ED", zorder=0)
                ax.spines[["top", "right", "left"]].set_visible(False)
                ax.tick_params(axis="y", length=0)
                ax.bar_label(bars, labels=[f"{value:.{precision}f}" for value in values], padding=7,
                             fontsize=11, fontweight="bold", color="#19364F")
            fig.text(.5, .035, note, ha="center", fontsize=10, color="#526271")
            fig.tight_layout(rect=(0, .09, 1, .9), w_pad=3)
            fig.savefig(output, dpi=200, facecolor="white")
        finally:
            plt.close(fig)


def create_report(data: ReportData, output_dir: Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / name for name in (
        "validation_k.png", "validation_gamma.png", "test_comparison.png",
        "results.csv", "execution_results.txt",
        "execution_results.png",
    )]
    selected_k, selected_gamma = data.adjusted["top_k"], data.adjusted["gamma"]
    users = data.k_results[0]["num_users"]
    _plot_pair(data.k_results, [str(r["top_k"]) for r in data.k_results],
               "Validation: neighbor count K (gamma = 0)",
               f"Validation users: {users:,} | Highlighted: selected K = {selected_k}",
               [r["top_k"] == selected_k for r in data.k_results], paths[0])
    _plot_pair(data.gamma_results, [f"{r['gamma']:g}" for r in data.gamma_results],
               f"Validation: popularity adjustment gamma (K = {selected_k})",
               f"Validation users: {users:,} | Highlighted: selected gamma = {selected_gamma:g}",
               [r["gamma"] == selected_gamma for r in data.gamma_results], paths[1])
    _plot_pair([data.baseline, data.adjusted], ["Before", "After"],
               f"Test: before vs. after adjustment (K = {selected_k})",
               f"Test users: {data.baseline['num_users']:,} | gamma: 0 vs. {selected_gamma:g}",
               [False, True], paths[2])
    with paths[3].open("w", encoding="utf-8-sig", newline="") as file:
        fields = ["experiment", "split", "top_k", "gamma", "num_users", "recall_at_10", "arp"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for experiment, results in (("validation_k", data.k_results),
                                    ("validation_gamma", data.gamma_results),
                                    ("test_baseline", [data.baseline]),
                                    ("test_adjusted", [data.adjusted])):
            for result in results:
                writer.writerow({"experiment": experiment, **{key: result[key] for key in fields[1:]}})
    paths[4].write_text(format_execution_results(data), encoding="utf-8")
    # 저장된 수치 요약을 이미지로도 제공한다. 원래 훈련 터미널 캡처는 아니다.
    from matplotlib import pyplot as plt

    body = format_execution_results(data).split("\nInput files:")[0]
    fig = plt.figure(figsize=(13, 9), facecolor="#10283D")
    try:
        fig.text(.045, .95, "Execution results summary", fontsize=21, fontweight="bold", color="white")
        fig.text(.045, .895, body, va="top", fontsize=11, fontfamily="DejaVu Sans Mono",
                 linespacing=1.6, color="#E7EEF3")
        fig.savefig(paths[5], dpi=200, facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="저장된 결과로 그래프와 실행 결과 요약 생성 (훈련 없음)")
    parser.add_argument("--results-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/report"))
    parser.add_argument("--k-values", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0, .1, .25])
    parser.add_argument("--selected-k", type=int, default=20)
    args = parser.parse_args()
    try:
        data = load_report_data(args.results_dir, args.k_values, args.gammas, args.selected_k)
        paths = create_report(data, args.output_dir)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"출력 생성 실패: {exc}\n")
    print(format_execution_results(data))
    print("생성 파일:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
