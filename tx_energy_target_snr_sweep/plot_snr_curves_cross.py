#!/usr/bin/env python3
"""Plot ZF vs VPP BER/BLER vs SNR(dB) for the 'cross' numerology combos:
32x32 with 96 subcarriers (fft_size=128) and 8x8/16x16 with 1216
subcarriers (fft_size=2048) -- i.e. swapped relative to the original
snr_experiment_results.json, where 8x8/16x16 used 96 subcarriers and
32x32 used 1216.

Data collected by snr_experiment_cross.py (see that file for exact
per-config tx_energy_target, noise_level and SNR_dB formula -- same
convention as plot_snr_curves.py's docstring: SNR_dB =
10*log10(target/(BsAntNum*OfdmDataNum) / noise_level^2)).

tx_energy_target used per config (NOT all "natural" -- 32x32 at 96
subcarriers only stays error-free up to ~0.2x its own natural ZF energy,
confirmed separately via a zero-noise ratio sweep, see
tx_energy_target_zeronoise_sweep/zeronoise_32x32_sc96.json):
  8x8_sc1216:   target = 15948.0 (= natural)
  16x16_sc1216: target = 27896.0 (= natural)
  32x32_sc96:   target = 1675.9  (= 0.2 x natural)
"""
import json
import statistics
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_PATH = "snr_experiment_cross_results.json"
OUTPUT_PATH = "zf_vs_vpp_snr_cross.png"

COLOR_ZF = "#0072B2"   # Okabe-Ito blue
COLOR_VPP = "#D55E00"  # Okabe-Ito vermillion
FLOOR = 2e-7           # display floor for exactly-0 measurements on log scale

SIZES = ["8x8_sc1216", "16x16_sc1216", "32x32_sc96"]


def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def main():
    data = json.load(open(RESULTS_PATH))

    sizes = SIZES
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


if __name__ == "__main__":
    main()
