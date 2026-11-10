#!/usr/bin/env python3
"""Same data as snr_experiment_results.json / plot_snr_curves.py, split into
two figures by OFDM numerology instead of one 3-column figure:

  - fft_size=2048, ofdm_data_num=1216  -> only the 32x32 run used this
  - fft_size=128,  ofdm_data_num=96    -> the 8x8 and 16x16 runs used this

No new simulation runs: this just re-groups and re-plots the existing
snr_experiment_results.json (see plot_snr_curves.py's docstring for how
that data was collected and what SNR_dB means).
"""
import json
import statistics
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_PATH = "snr_experiment_results.json"

COLOR_ZF = "#0072B2"   # Okabe-Ito blue
COLOR_VPP = "#D55E00"  # Okabe-Ito vermillion
FLOOR = 2e-7           # display floor for exactly-0 measurements on log scale

GROUPS = [
    ("zf_vs_vpp_snr_fft2048_ofdm1216.png", ["32x32"]),
    ("zf_vs_vpp_snr_fft128_ofdm96.png", ["8x8", "16x16"]),
]


def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def plot_group(data, sizes, OUTPUT_PATH):
    metrics = [("zf_ber", "vpp_ber", "BER"), ("zf_bler", "vpp_bler", "BLER")]

    fig, axes = plt.subplots(2, len(sizes), figsize=(5 * len(sizes), 5),
                              sharex=False, squeeze=False)

    for col, size in enumerate(sizes):
        dbs = sorted(data[size], key=lambda x: int(x))
        db_vals = [int(x) for x in dbs]
        for row, (zf_key, vpp_key, label) in enumerate(metrics):
            ax = axes[row][col]
            zf_vals = [max(mean_or_none(data[size][db][zf_key]), FLOOR) for db in dbs]
            vpp_vals = [max(mean_or_none(data[size][db][vpp_key]), FLOOR) for db in dbs]
            ax.plot(db_vals, zf_vals, "-o", color=COLOR_ZF, label="ZF",
                    linewidth=2, markersize=5)
            ax.plot(db_vals, vpp_vals, "-o", color=COLOR_VPP, label="VPP",
                    linewidth=2, markersize=5)
            ax.set_yscale("log")
            ax.set_ylim(FLOOR * 0.5, 1.5)
            ax.grid(True, which="both", linestyle=":", linewidth=0.5,
                    color="#cccccc", alpha=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row == 0:
                ax.set_title(size, fontsize=12, fontweight="bold")
            if row == 1:
                ax.set_xlabel("SNR", fontsize=11)
            if col == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold")
            ax.axhline(FLOOR, color="#999999", linewidth=0.5, linestyle="--",
                       alpha=0.5)

    # 整张图共用一个 legend，放在最底部（x 轴标签下面）
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.0))

    # 底部留出 legend 的空间；没有 suptitle，顶部不用再留白
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"saved {OUTPUT_PATH}")


def main():
    data = json.load(open(RESULTS_PATH))
    for output_path, sizes in GROUPS:
        plot_group(data, sizes, output_path)


if __name__ == "__main__":
    main()
