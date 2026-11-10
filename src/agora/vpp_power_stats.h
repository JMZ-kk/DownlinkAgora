/**
 * @file vpp_power_stats.h
 * @brief Per-OFDM-symbol transmit-energy bookkeeping for Vpp power
 * normalization.
 *
 * Vpp lowers the energy of the transmitted vector P*(u+tau*l) relative to
 * plain ZF's P*u, but that only helps if the saved energy is spent on a
 * larger gain (otherwise the receiver, whose noise is fixed, sees the same
 * signal power and only pays the modulo loss). Each DoPrecode block task
 * records, for its block of subcarriers, the energy plain ZF would have sent
 * (e_ref) and the energy actually sent (e_vpp). DoIFFT then scales the whole
 * symbol by sqrt(sum e_ref / sum e_vpp), so a Vpp symbol carries the same
 * total energy a ZF symbol would have. The UE recovers that gain from the
 * (unperturbed, known) DMRS tones of the same symbol.
 */
#ifndef VPP_POWER_STATS_H_
#define VPP_POWER_STATS_H_

#include <cmath>
#include <cstddef>
#include <vector>

class VppPowerStats {
 public:
  VppPowerStats(size_t frame_window, size_t dl_syms_per_frame,
                size_t blocks_per_symbol)
      : dl_syms_(dl_syms_per_frame),
        blocks_(blocks_per_symbol),
        frame_window_(frame_window),
        e_ref_(frame_window * dl_syms_per_frame * blocks_per_symbol, 0.0),
        e_vpp_(frame_window * dl_syms_per_frame * blocks_per_symbol, 0.0) {}

  // Each (frame slot, dl symbol, block) is written by exactly one precode
  // task, so no locking is needed; the precode->IFFT scheduling handoff
  // provides the ordering.
  void Set(size_t frame_id, size_t dl_symbol_idx, size_t block_idx,
           double e_ref, double e_vpp) {
    const size_t i = Index(frame_id, dl_symbol_idx, block_idx);
    e_ref_[i] = e_ref;
    e_vpp_[i] = e_vpp;
  }


  // Amplitude gain sqrt(sum e_ref / sum e_vpp) for one DL symbol.
  double Gain(size_t frame_id, size_t dl_symbol_idx) const {
    double ref = 0.0;
    double vpp = 0.0;
    for (size_t b = 0; b < blocks_; b++) {
      const size_t i = Index(frame_id, dl_symbol_idx, b);
      ref += e_ref_[i];
      vpp += e_vpp_[i];
    }
    const double gain = (vpp > 0.0) ? std::sqrt(ref / vpp) : 1.0;
    return std::isfinite(gain) ? gain : 1.0;
  }

  // The actual precoded energy sum(e_vpp) for one DL symbol -- ||P*u||^2 for
  // Zf, ||P*(u+tau*l)||^2 for Vpp/Se/Fcse (e_ref is unused here). For
  // Config::TxEnergyTarget(): amplitude gain sqrt(target / sum e_vpp) that
  // rescales THIS symbol's own energy to an external constant, independent
  // of what the other scheme (or this one, on a different frame/subcarrier)
  // would have used.
  double GainToTarget(size_t frame_id, size_t dl_symbol_idx,
                      double target) const {
    double vpp = 0.0;
    for (size_t b = 0; b < blocks_; b++) {
      vpp += e_vpp_[Index(frame_id, dl_symbol_idx, b)];
    }
    const double gain = (vpp > 0.0) ? std::sqrt(target / vpp) : 1.0;
    return std::isfinite(gain) ? gain : 1.0;
  }

 private:
  size_t Index(size_t frame_id, size_t dl_symbol_idx, size_t block_idx) const {
    return (((frame_id % frame_window_) * dl_syms_ + dl_symbol_idx) * blocks_) +
           block_idx;
  }

  size_t dl_syms_;
  size_t blocks_;
  size_t frame_window_;
  std::vector<double> e_ref_;
  std::vector<double> e_vpp_;
};

#endif  // VPP_POWER_STATS_H_
