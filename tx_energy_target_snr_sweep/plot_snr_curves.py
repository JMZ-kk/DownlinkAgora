#!/usr/bin/env python3
"""Plot ZF vs VPP BER/BLER vs SNR(dB) from snr_experiment_results.json.

Both algorithms are rescaled, per DL data symbol, to the SAME fixed
transmit energy via Agora's `tx_energy_target` config knob (see
src/agora/vpp_power_stats.h:GainToTarget and src/agora/doifft.cc). SNR(dB)
is defined as:

    SNR_dB = 10 * log10( E_per_sample / noise_level^2 )
    E_per_sample = tx_energy_target / (BsAntNum * OfdmDataNum)

i.e. the average per-antenna, per-data-subcarrier transmit energy (after
rescaling to the fixed target) divided by the AWGN noise variance. This is
NOT the true post-channel receive SNR (that depends on the realized H and
the ZF precoder's own normalization, which this metric deliberately
excludes) -- it's a clean, reproducible reference scale that depends only
on tx_energy_target and noise_level, not on the random payload data or
which algorithm (ZF/VPP) is active. See the per-size E_per_sample below
for how it was derived.

Data collection (see snr_experiment.py in this same directory): for each
(MIMO size, SNR dB point), noise_level was solved from the formula above,
then ZF and VPP were each run 5 times (max_frame=20 per run, channel_seed
fixed) and BER/BLER were averaged across the 5 reps.
"""
import json
import statistics
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_PATH = "snr_experiment_results.json"
OUTPUT_PATH = "zf_vs_vpp_snr_chart.png"

# tx_energy_target and E_per_sample = target / (BsAntNum * OfdmDataNum) used
# for each MIMO size -- see the module docstring for the SNR_dB formula.
SIZE_INFO = {
    "8x8": dict(target=1158.0, bsant=8, ofdm=96),
    "16x16": dict(target=1074.5, bsant=16, ofdm=96),
    "32x32": dict(target=101316.0, bsant=32, ofdm=1216),
}

COLOR_ZF = "#0072B2"   # Okabe-Ito blue
COLOR_VPP = "#D55E00"  # Okabe-Ito vermillion
FLOOR = 2e-7           # display floor for exactly-0 measurements on log scale


def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def main():
    data = json.load(open(RESULTS_PATH))

    sizes = ["8x8", "16x16", "32x32"]
    metrics = [("zf_ber", "vpp_ber", "BER"), ("zf_bler", "vpp_bler", "BLER")]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False)

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
                ax.set_title(f"{size} (same transmit energy)", fontsize=12,
                             fontweight="bold")
            if row == 1:
                ax.set_xlabel("SNR (dB)\n= 10*log10(E_per_sample / noise_level^2)",
                               fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold")
            if row == 0 and col == 2:
                ax.legend(loc="lower left", frameon=False, fontsize=10)
            ax.axhline(FLOOR, color="#999999", linewidth=0.5, linestyle="--",
                       alpha=0.5)

    fig.suptitle(
        "ZF vs VPP, both rescaled to the same fixed per-symbol transmit energy "
        "(TxEnergyTarget)\n"
        "(points at the dashed floor line = exactly 0 errors measured, "
        "5 reps x max_frame=20 each)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
