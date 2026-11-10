/**
 * @file doprecode.h
 * @brief Declaration file for the DoPrecode class.
 */
#ifndef DOPRECODE_H_
#define DOPRECODE_H_

#include <cstdint>
#include <iostream>
#include <vector>

#include "common_typedef_sdk.h"
#include "config.h"
#include "doer.h"
#include "mac_scheduler.h"
#include "memory_manage.h"
#include "message.h"
#include "mkl_dfti.h"
#include "stats.h"
#include "symbols.h"
#include "vpp_power_stats.h"

class DoPrecode : public Doer {
 public:
  DoPrecode(Config* in_config, int in_tid,
            PtrGrid<kFrameWnd, kMaxDataSCs, complex_float>& dl_beam_matrices_,
            Table<complex_float>& in_dl_ifft_buffer,
            Table<int8_t>& dl_encoded_or_raw_data, MacScheduler* mac_sched,
            Stats* in_stats_manager, VppPowerStats& vpp_power);
  ~DoPrecode() override;

  /**
   * Do demodulation task for a block of subcarriers (demul_block_size)
   * @param tid: task thread index, used for selecting spm_buffer and task
   * ptok
   * @param offset: offset of the first subcarrier in the block in
   * data_buffer_ Buffers: data_buffer_, spm_buffer_, precoder_buffer_,
   * equal_buffer_, demul_hard_buffer_ Input buffer: data_buffer_,
   * precoder_buffer_ Output buffer: demul_hard_buffer_ Intermediate buffer:
   * spm_buffer, equal_buffer_ Offsets: data_buffer_: dim1: frame index * # of
   * data symbols per frame + data symbol index dim2: transpose block
   * index * block size * # of antennas + antenna index * block size
   *     spm_buffer:
   *         dim1: task thread index
   *         dim2: antenna index
   *     precoder_buffer_:
   *         dim1: frame index * FFT size + subcarrier index in the current
   * frame equal_buffer_, demul_buffer: dim1: frame index * # of data
   * symbols per frame + data symbol index dim2: subcarrier index * # of
   * users Event offset: offset Description:
   *     1. for each subcarrier in the block, block-wisely copy data from
   * data_buffer_ to spm_buffer_
   *     2. perform equalization with data and percoder matrixes
   *     3. perform demodulation on equalized data matrix
   *     4. add an event to the message queue to infrom main thread the
   * completion of this task
   */
  EventData Launch(size_t tag) override;

  // Load input data for a single UE and a single subcarrier
  void LoadInputData(size_t frame_id, size_t symbol_idx_dl, size_t sp_id,
                     size_t user_id, size_t sc_id, size_t sc_id_in_block);
  // `perturb` must be false for DL pilot symbols and non-data (DMRS)
  // subcarriers: the UE estimates its channel / tracks phase from those, so
  // they must reach it as exactly the known pilot values (VPP/VPP only).
  // `vpp_tau`/`vpp_window` are the current modulation order's perturbation
  // spacing/search-window half-width, computed once per Launch() call (see
  // doprecode.cc) instead of once per subcarrier -- both only depend on the
  // DL modulation order, constant for the whole block.
  void PrecodingPerSc(size_t frame_id, size_t frame_slot, size_t sc_id,
                      size_t sc_id_in_block, size_t ue_num, bool perturb,
                      double vpp_tau, int vpp_window);

 private:
  // True only for real data symbols on data subcarriers (see PrecodingPerSc).
  bool IsPerturbable(size_t symbol_idx_dl, size_t sc_id) const;

  // Vpp = VPP + LLL lattice reduction (as a precondition, not an optional
  // optimization) + Babai nearest-plane centering, per QUIOVPPDOWNLINK's
  // paper recipe (paper/design.tex). Only called when
  // CommsLib::IsPerturbationPrecoder(cfg_->BeamformingAlgo()); VPP, SE and
  // FCSE share everything here except the integer solver that is called.
  // `beam_sc_id` is cfg_->GetBeamScId(sc_id) -- under freq_orthogonal_pilot
  // this collapses groups of PilotScGroupSize() subcarriers to the same
  // value, which is how the LLL-reduction cache below (vpp_lll_cache_*)
  // knows when it's safe to reuse the previous subcarrier's result.
  void ApplyVppPerturbation(size_t frame_id, size_t beam_sc_id,
                             const arma::cx_fmat& mat_precoder,
                             arma::cx_fmat& mat_data, size_t ue_num,
                             double tau, int window);


  PtrGrid<kFrameWnd, kMaxDataSCs, complex_float>& dl_beam_matrices_;
  Table<complex_float>& dl_ifft_buffer_;
  Table<int8_t>& dl_raw_data_;
  MacScheduler* mac_sched_;
  Table<float> qam_table_;
  DurationStat* duration_stat_;
  complex_float* modulated_buffer_temp_;
  complex_float* precoded_buffer_temp_;

  // Vpp power normalization (see vpp_power_stats.h). Accumulated over
  // the subcarriers of the current Launch() block and published at its end.
  VppPowerStats& vpp_power_;
  bool track_power_ = false;
  double block_e_ref_ = 0.0;
  double block_e_vpp_ = 0.0;

  // Caches the channel-only (data-independent) part of VPP's LLL/Babai
  // preconditioning -- (tmap, q_red, sc) -- across subcarriers that share
  // the same beam_sc_id under freq_orthogonal_pilot (a no-op otherwise,
  // since beam_sc_id == sc_id there and never repeats within one Launch()
  // call's block). Safe as plain members, no atomics/locking: Launch()'s
  // subcarrier loop runs sequentially on this single thread, so there is
  // never a concurrent writer -- unlike an earlier, reverted attempt to
  // share this cache ACROSS a frame's independently-scheduled DL symbols
  // (see project-vpp-full-load-findings memory), this one never crosses a
  // task/thread boundary.
  bool vpp_lll_cache_valid_ = false;
  size_t vpp_lll_cache_frame_id_ = 0;
  size_t vpp_lll_cache_beam_sc_id_ = 0;
  size_t vpp_lll_cache_num_int_ = 0;
  arma::mat vpp_lll_cache_tmap_;
  arma::mat vpp_lll_cache_q_red_;
  double vpp_lll_cache_sc_ = 1.0;
  bool vpp_lll_cache_reduced_ = true;
  // chol(q_red, "upper"), also channel-only (independent of the per-symbol
  // data vector b_red) -- cached alongside q_red/tmap/sc so the per-SC Babai
  // centering step reuses it via triangular solves instead of re-factoring
  // q_red and re-solving it generically on every subcarrier in the group.
  arma::mat vpp_lll_cache_rr_;
};

#endif  // DOPRECODE_H_
