/**
 * @file vpp_utils.h
 * @brief Shared BS/UE helpers for Vector Perturbation Precoding (VPP):
 * the perturbation lattice spacing (tau) and the receiver-side modulo fold
 * that undoes it. Ported from the lattice-VPP formulation in
 * QUIOVPPDOWNLINK/vpp_qubo/modulation_utils.py (conventional_tau, gamma_tau).
 */
#ifndef VPP_UTILS_H_
#define VPP_UTILS_H_

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "comms-lib.h"
#include "common_typedef_sdk.h"

/**
 * @brief Perturbation lattice spacing tau for a square Gray-mapped QAM
 * constellation with the given number of bits per symbol, using Agora's own
 * per-modulation power normalization (see modulation.cc: scale =
 * 1/sqrt(2*(M-1)/3)). tau = 2*(c_max + delta/2) * scale, matching the
 * "conventional_tau" formula in modulation_utils.py, generalized to any
 * square M-QAM order and rescaled into Agora's normalized symbol units.
 */
inline double ComputeVppTau(CommsLib::ModulationOrder mod_order) {
  const size_t mod_order_bits = static_cast<size_t>(mod_order);
  const size_t m = static_cast<size_t>(1) << mod_order_bits;  // constellation size
  const size_t levels_per_dim = static_cast<size_t>(1)
                                 << (mod_order_bits / 2);  // sqrt(M)
  const double cmax_raw = static_cast<double>(levels_per_dim) - 1.0;
  const double delta_raw = 2.0;  // Gray PAM levels are always spaced by 2
  const double scale = 1.0 / std::sqrt(2.0 * (static_cast<double>(m) - 1.0) / 3.0);
  return 2.0 * (cmax_raw + 0.5 * delta_raw) * scale;
}

/**
 * @brief Default half-width of the Vpp search window in reduced coordinates:
 * the paper's "bound + 2" (red_window in 7_8_26_intmaxbm_downlink.cpp), where
 * bound = sqrt(M) - 1 is the largest raw PAM level -- 3 for QPSK, 5 for 16QAM,
 * 9 for 64QAM.
 */
inline int DefaultVppLllWindow(CommsLib::ModulationOrder mod_order) {
  const size_t mod_order_bits = static_cast<size_t>(mod_order);
  const int levels_per_dim = 1 << (mod_order_bits / 2);
  return (levels_per_dim - 1) + 2;
}

/**
 * @brief Receiver-side modulo-tau fold: undoes the integer-lattice
 * perturbation VPP added at the transmitter by wrapping each real/imaginary
 * component back into (-tau/2, tau/2]. Must be applied to the equalized
 * downlink symbol *before* demodulation whenever beamforming is "VPP", "SE"
 * or "FCSE".
 */
/**
 * @brief Makes the soft-demodulator's MSB LLRs periodic in tau after the
 * modulo fold. With Agora's Gray maps the two extreme levels of an axis
 * differ only in the MSB, and they are neighbours across the fold boundary
 * +-tau/2, so a sample near +-tau/2 is ambiguous between them -- yet the
 * stock demapper's MSB LLR is linear in x and is most confident exactly there.
 * Replaces each MSB LLR by the same slope times a triangle wave of period tau
 * (unchanged for |x| <= tau/4, falling to 0 at +-tau/2); the other bits never
 * flip across the fold and are left alone. `llr` holds mod_bits bytes per
 * symbol with the real/imag MSB LLRs first, as Demodulate() writes them for
 * QPSK/16QAM/64QAM; other orders are left untouched.
 */
inline void ApplyPeriodicMsbLlr(const complex_float* folded, int8_t* llr,
                                size_t num, size_t mod_bits, double tau) {
  if (mod_bits != 2 && mod_bits != 4 && mod_bits != 6) {
    return;
  }
  const double quarter = tau / 4.0;
  const double half = tau / 2.0;
  for (size_t i = 0; i < num; i++) {
    const float axes[2] = {folded[i].re, folded[i].im};
    for (size_t a = 0; a < 2; a++) {
      const double x = axes[a];
      if (std::fabs(x) <= quarter) {
        continue;
      }
      const double tri = std::copysign(half - std::fabs(x), x);
      int8_t& b = llr[i * mod_bits + a];
      b = static_cast<int8_t>(std::lround(b * tri / x));
    }
  }
}

inline complex_float GammaTau(complex_float x, double tau) {
  auto gamma = [tau](float r) -> float {
    return static_cast<float>(r - tau * std::floor(static_cast<double>(r) / tau + 0.5));
  };
  return {gamma(x.re), gamma(x.im)};
}

#endif  // VPP_UTILS_H_
