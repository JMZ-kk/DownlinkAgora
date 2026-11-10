/**
 * @file rayleigh_model.h
 * @brief Declaration file for the rayleigh channel model
 */
#ifndef RAYLEIGH_MODEL_H_
#define RAYLEIGH_MODEL_H_

#include <cstdint>
#include <random>

#include "armadillo"
#include "channel_model.h"

//Flat fading Rayleigh channel model
class RayleighModel : public ChannelModel {
 public:
  // seed == 0 keeps the default behaviour (Armadillo's global RNG, shared with
  // the AWGN noise draws). A nonzero seed draws the per-frame channel matrices
  // from a private generator instead, so repeated runs see the same channel
  // sequence -- the "channel_seed" config option, for paired A/B comparisons
  // of downlink algorithms.
  RayleighModel(size_t bs_ant_num, size_t ue_ant_num, size_t samples_per_sym,
                uint64_t seed = 0)
      : ChannelModel(bs_ant_num, ue_ant_num, samples_per_sym,
                     ChannelModel::kFlat),
        seeded_(seed != 0),
        rng_(seed) {}

  void UpdateModel() final {
    arma::fmat rmat(ues_num_, bss_num_);
    arma::fmat imat(ues_num_, bss_num_);
    if (seeded_) {
      std::normal_distribution<float> normal(0.0f, 1.0f);
      for (auto& v : rmat) {
        v = normal(rng_);
      }
      for (auto& v : imat) {
        v = normal(rng_);
      }
    } else {
      rmat.randn();
      imat.randn();
    }

    h_flat_ = arma::cx_fmat(rmat, imat);
  }

 private:
  const bool seeded_;
  std::mt19937_64 rng_;
};

#endif  // RAYLEIGH_MODEL_H_
