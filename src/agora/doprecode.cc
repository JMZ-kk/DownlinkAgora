/**
 * @file doprecode.cc
 * @brief Implementation file for the DoPrecode class.
 */
#include "doprecode.h"

#include <limits>

#include "comms-lib.h"
#include "concurrent_queue_wrapper.h"
#include "logger.h"
#include "modulation.h"
#include "vpp_solver.h"
#include "vpp_utils.h"

static constexpr bool kUseSpatialLocality = true;

DoPrecode::DoPrecode(
    Config* in_config, int in_tid,
    PtrGrid<kFrameWnd, kMaxDataSCs, complex_float>& dl_beam_matrices,
    Table<complex_float>& in_dl_ifft_buffer,
    Table<int8_t>& dl_encoded_or_raw_data /* Encoded if LDPC is enabled */,
    MacScheduler* mac_sched, Stats* in_stats_manager,
    VppPowerStats& vpp_power)
    : Doer(in_config, in_tid),
      dl_beam_matrices_(dl_beam_matrices),
      dl_ifft_buffer_(in_dl_ifft_buffer),
      dl_raw_data_(dl_encoded_or_raw_data),
      mac_sched_(mac_sched),
      vpp_power_(vpp_power) {
  duration_stat_ =
      in_stats_manager->GetDurationStat(DoerType::kPrecode, in_tid);

  AllocBuffer1d(&modulated_buffer_temp_, kSCsPerCacheline * cfg_->UeAntNum(),
                Agora_memory::Alignment_t::kAlign64, 0);
  AllocBuffer1d(&precoded_buffer_temp_,
                cfg_->DemulBlockSize() * cfg_->BsAntNum(),
                Agora_memory::Alignment_t::kAlign64, 0);
}

DoPrecode::~DoPrecode() {
  FreeBuffer1d(&modulated_buffer_temp_);
  FreeBuffer1d(&precoded_buffer_temp_);
}

EventData DoPrecode::Launch(size_t tag) {
  size_t start_tsc = GetTime::WorkerRdtsc();
  const size_t frame_id = gen_tag_t(tag).frame_id_;
  const size_t base_sc_id = gen_tag_t(tag).sc_id_;
  const size_t symbol_id = gen_tag_t(tag).symbol_id_;
  const size_t symbol_idx_dl = cfg_->Frame().GetDLSymbolIdx(symbol_id);
  const size_t total_symbol_idx =
      cfg_->GetTotalSymbolIdxDl(frame_id, symbol_idx_dl);
  const size_t frame_slot = frame_id % kFrameWnd;

  if (kDebugPrintInTask) {
    std::printf(
        "In doPrecode thread %d: frame %zu, symbol %zu, subcarrier %zu\n", tid_,
        frame_id, symbol_id, base_sc_id);
  }

  size_t max_sc_ite =
      std::min(cfg_->DemulBlockSize(), cfg_->OfdmDataNum() - base_sc_id);

  auto ue_list = mac_sched_->ScheduledUeList(frame_id, base_sc_id);

  // Vpp power normalization: only real DL data symbols are rescaled
  // (pilot symbols must reach the UE at their nominal power). Also tracked
  // unconditionally (any algorithm, not just perturbation precoders) when
  // Config::TxEnergyTarget() is set: see doifft.cc -- that mode rescales
  // every algorithm's own actual precoded energy (||P*u||^2 for Zf,
  // ||P*(u+tau*l)||^2 for Vpp) to one external, P/u-independent constant,
  // instead of matching Vpp to whatever Zf's own energy happens to be.
  track_power_ =
      ((cfg_->VppPowerNorm() &&
        CommsLib::IsPerturbationPrecoder(cfg_->BeamformingAlgo())) ||
       cfg_->TxEnergyTarget() > 0.0) &&
      (symbol_idx_dl >= cfg_->Frame().ClientDlPilotSymbols());
  block_e_ref_ = 0.0;
  block_e_vpp_ = 0.0;

  // Hoisted out of the per-subcarrier loop below: both only depend on the
  // DL modulation order, which is constant for this whole block (same
  // mac_sched_ group as ue_list above), so there is no reason to recompute
  // them on every one of the up to demul_block_size subcarriers.
  double vpp_tau = 0.0;
  int vpp_window = 0;
  if (CommsLib::IsPerturbationPrecoder(cfg_->BeamformingAlgo())) {
    const auto mod_order = static_cast<CommsLib::ModulationOrder>(
        mac_sched_->Params().ModOrderBits(Direction::kDownlink));
    vpp_tau = ComputeVppTau(mod_order);
    vpp_window = cfg_->VppLllWindow() > 0 ? cfg_->VppLllWindow()
                                          : DefaultVppLllWindow(mod_order);
  }

  if (kUseSpatialLocality) {
    for (size_t i = 0; i < max_sc_ite; i = i + kSCsPerCacheline) {
      size_t start_tsc1 = GetTime::WorkerRdtsc();
      for (size_t sp_id = 0; sp_id < ue_list.n_elem; sp_id++) {
        for (size_t j = 0; j < kSCsPerCacheline; j++) {
          LoadInputData(frame_id, symbol_idx_dl, sp_id, ue_list.at(sp_id),
                        base_sc_id + i + j, j);
        }
      }

      size_t start_tsc2 = GetTime::WorkerRdtsc();
      duration_stat_->task_duration_[1] += start_tsc2 - start_tsc1;
      for (size_t j = 0; j < kSCsPerCacheline; j++) {
        PrecodingPerSc(frame_id, frame_slot, base_sc_id + i + j, i + j,
                       ue_list.n_elem,
                       IsPerturbable(symbol_idx_dl, base_sc_id + i + j),
                       vpp_tau, vpp_window);
      }
      duration_stat_->task_count_ =
          duration_stat_->task_count_ + kSCsPerCacheline;
      duration_stat_->task_duration_[2] += GetTime::WorkerRdtsc() - start_tsc2;
    }
  } else {
    for (size_t i = 0; i < max_sc_ite; i++) {
      size_t start_tsc1 = GetTime::WorkerRdtsc();
      int cur_sc_id = base_sc_id + i;
      for (size_t sp_id = 0; sp_id < ue_list.n_elem; sp_id++) {
        LoadInputData(frame_id, symbol_idx_dl, sp_id, ue_list.at(sp_id),
                      cur_sc_id, 0);
      }
      size_t start_tsc2 = GetTime::WorkerRdtsc();
      duration_stat_->task_duration_[1] += start_tsc2 - start_tsc1;

      PrecodingPerSc(frame_id, frame_slot, cur_sc_id, i, ue_list.n_elem,
                     IsPerturbable(symbol_idx_dl, cur_sc_id), vpp_tau,
                     vpp_window);
      duration_stat_->task_count_++;
      duration_stat_->task_duration_[2] += GetTime::WorkerRdtsc() - start_tsc2;
    }
  }

  size_t start_tsc3 = GetTime::WorkerRdtsc();

  __m256i index = _mm256_setr_epi64x(0, cfg_->BsAntNum(), cfg_->BsAntNum() * 2,
                                     cfg_->BsAntNum() * 3);
  auto* precoded_ptr = reinterpret_cast<float*>(precoded_buffer_temp_);
  for (size_t ant_id = 0; ant_id < cfg_->BsAntNum(); ant_id++) {
    int ifft_buffer_offset = ant_id + cfg_->BsAntNum() * total_symbol_idx;
    auto* ifft_ptr = reinterpret_cast<float*>(
        &dl_ifft_buffer_[ifft_buffer_offset]
                        [base_sc_id + cfg_->OfdmDataStart()]);
    for (size_t i = 0; i < cfg_->DemulBlockSize() / 4; i++) {
      float* input_shifted_ptr =
          precoded_ptr + 4 * i * 2 * cfg_->BsAntNum() + ant_id * 2;
      __m256d t_data = _mm256_i64gather_pd(
          reinterpret_cast<double*>(input_shifted_ptr), index, 8);
      _mm256_stream_pd(reinterpret_cast<double*>(ifft_ptr + i * 8), t_data);
    }
  }
  duration_stat_->task_duration_[3] += GetTime::WorkerRdtsc() - start_tsc3;
  duration_stat_->task_duration_[0] += GetTime::WorkerRdtsc() - start_tsc;
  if (kDebugPrintInTask) {
    std::printf(
        "In doPrecode thread %d: finished frame: %zu, symbol: %zu, "
        "subcarrier: %zu\n",
        tid_, frame_id, symbol_id, base_sc_id);
  }
  if (track_power_) {
    vpp_power_.Set(frame_id, symbol_idx_dl, base_sc_id / cfg_->DemulBlockSize(),
                   block_e_ref_, block_e_vpp_);
  }
  return {EventType::kPrecode, tag};
}

void DoPrecode::LoadInputData(size_t frame_id, size_t symbol_idx_dl,
                              size_t sp_id, size_t user_id, size_t sc_id,
                              size_t sc_id_in_block) {
  complex_float* data_ptr =
      modulated_buffer_temp_ + sc_id_in_block * cfg_->UeAntNum();
  if ((symbol_idx_dl < cfg_->Frame().ClientDlPilotSymbols()) ||
      (cfg_->IsDataSubcarrier(sc_id) == false)) {
    data_ptr[sp_id] = cfg_->UeSpecificPilot()[user_id][sc_id];
  } else {
    size_t data_symbol_idx_dl =
        symbol_idx_dl - cfg_->Frame().ClientDlPilotSymbols();
    int8_t* raw_data_ptr = cfg_->GetModBitsBuf(
        kDebugBypassEncode ? cfg_->DlModBits() : dl_raw_data_,
        Direction::kDownlink, kDebugBypassEncode ? 0 : frame_id,
        data_symbol_idx_dl, user_id, cfg_->GetOFDMDataIndex(sc_id));
    Table<complex_float> mod_table =
        mac_sched_->Params().ModTable(Direction::kDownlink);
    data_ptr[sp_id] = ModSingleUint8((uint8_t)(*raw_data_ptr), mod_table);
  }
}

// || P * d ||^2 for a column-major (bs_ant x ue_num) precoder P and a
// length-ue_num data vector d.
static double PrecodedEnergy(const arma::cx_float* p, const arma::cx_float* d,
                             size_t bs_ant, size_t ue_num) {
  double e = 0.0;
  for (size_t a = 0; a < bs_ant; a++) {
    arma::cx_float acc(0.0f, 0.0f);
    for (size_t k = 0; k < ue_num; k++) {
      acc += p[k * bs_ant + a] * d[k];
    }
    e += std::norm(acc);
  }
  return e;
}

bool DoPrecode::IsPerturbable(size_t symbol_idx_dl, size_t sc_id) const {
  return (symbol_idx_dl >= cfg_->Frame().ClientDlPilotSymbols()) &&
         cfg_->IsDataSubcarrier(sc_id);
}

void DoPrecode::PrecodingPerSc(size_t frame_id, size_t frame_slot,
                               size_t sc_id, size_t sc_id_in_block,
                               size_t ue_num, bool perturb, double vpp_tau,
                               int vpp_window) {
  const size_t beam_sc_id = cfg_->GetBeamScId(sc_id);
  arma::cx_float* precoder_ptr =
      reinterpret_cast<arma::cx_float*>(dl_beam_matrices_[frame_slot][beam_sc_id]);
  arma::cx_float* data_ptr = reinterpret_cast<arma::cx_float*>(
      modulated_buffer_temp_ +
      (kUseSpatialLocality
           ? (sc_id_in_block % kSCsPerCacheline * cfg_->UeAntNum())
           : 0));
  arma::cx_float* precoded_ptr = reinterpret_cast<arma::cx_float*>(
      precoded_buffer_temp_ + sc_id_in_block * cfg_->BsAntNum());

  if (track_power_) {
    block_e_ref_ += PrecodedEnergy(precoder_ptr, data_ptr, cfg_->BsAntNum(),
                                   ue_num);
  }

  if (perturb) {
    static std::atomic<int> dbg_calls{0};
    if (dbg_calls.fetch_add(1) < 3) {
      float max_data = 0.0f, max_precoder = 0.0f;
      for (size_t k = 0; k < ue_num; k++) {
        max_data = std::max(max_data, std::abs(data_ptr[k]));
      }
      for (size_t k = 0; k < ue_num * cfg_->BsAntNum(); k++) {
        max_precoder = std::max(max_precoder, std::abs(precoder_ptr[k]));
      }
      AGORA_LOG_ERROR(
          "DBGPROBE precode mod_bits=%zu sc=%zu max|data|=%f max|precoder|=%f "
          "ue_num=%zu bs_ant=%zu\n",
          mac_sched_->Params().ModOrderBits(Direction::kDownlink), sc_id,
          max_data, max_precoder, ue_num, cfg_->BsAntNum());
    }
  }

  if (perturb && CommsLib::IsPerturbationPrecoder(cfg_->BeamformingAlgo())) {
    arma::cx_fmat mat_precoder_vpp(precoder_ptr, cfg_->BsAntNum(), ue_num,
                                    false);
    arma::cx_fmat mat_data_vpp(data_ptr, ue_num, 1, false);
    ApplyVppPerturbation(frame_id, beam_sc_id, mat_precoder_vpp, mat_data_vpp,
                        ue_num, vpp_tau, vpp_window);
  }

#if defined(USE_MKL_CBLAS)
  MKL_Complex8 alpha = {1, 0};
  MKL_Complex8 beta = {0, 0};

  cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, cfg_->BsAntNum(), 1,
              ue_num, &alpha, (MKL_Complex8*)precoder_ptr, cfg_->BsAntNum(),
              (MKL_Complex8*)data_ptr, ue_num, &beta,
              (MKL_Complex8*)precoded_ptr, cfg_->BsAntNum());
#else
  arma::cx_fmat mat_precoder(precoder_ptr, cfg_->BsAntNum(), ue_num, false);
  arma::cx_fmat mat_data(data_ptr, ue_num, 1, false);
  arma::cx_fmat mat_precoded(precoded_ptr, cfg_->BsAntNum(), 1, false);
  mat_precoded = mat_precoder * mat_data;
  // cout << "Precoder: \n" << mat_precoder << endl;
  // cout << "Data: \n" << mat_data << endl;
  // cout << "Precoded data: \n" << mat_precoded << endl;
#endif

  if (track_power_) {
    double e = 0.0;
    for (size_t a = 0; a < cfg_->BsAntNum(); a++) {
      e += std::norm(precoded_ptr[a]);
    }
    block_e_vpp_ += e;
  }
}

// Vpp: x = P*(u + tau*l). P is already the ZF precoder (computed by
// DoBeamWeights, same formula as plain ZF -- see dobeamweights.cc); the
// integer search for l is preconditioned with LLL lattice reduction before
// searching, and centered on the Babai nearest-plane point instead of the
// origin -- per QUIOVPPDOWNLINK's paper (paper/design.tex Sec. 5.1/5.2).
// Reduction is what lets a single-coordinate annealer actually reach a good
// perturbation on the downlink's ZF-inverted (poorly conditioned) lattice;
// without it the search "rarely reaches the optimum" at a fixed budget
// (paper abstract).
//
// Recipe (mirrors dl_lll / generate_downlink_problem in
// QUIOVPPDOWNLINK/scripts/7_8_26_intmaxbm_downlink.cpp):
//   1. Q = tau^2 * G  (G = the realified Gram matrix of the ZF precoder)
//   2. basis = chol(Q), T = LllReduce(basis)   -- unimodular reduction
//   3. Q_red = T'*Q*T, b_red = T'*b
//   4. z_ctr = Babai nearest-plane point of Q_red/b_red (back-substitution
//      rounding on chol(Q_red)), then recenter: b_w = 2*Q_red*z_ctr + b_red
//   5. search a small window around w=0 (the Babai point) in reduced
//      coordinates for w minimizing w'*Q_red*w + b_w'*w
//   6. l = T*(z_ctr + w) maps back to the original perturbation coordinates
//
// Steps 1-3 only depend on the channel/precoder, not on the transmitted
// data (y) -- identical for all of this frame's DL data symbols at this
// subcarrier. Caching them ACROSS a frame's independently-scheduled DL
// symbols was tried and reverted (measured WORSE -- see
// project-vpp-full-load-findings memory): those symbols' Precode tasks run
// concurrently on different worker threads, so a cache shared between them
// needs cross-thread synchronization and mostly misses anyway.
//
// This cache is different: under freq_orthogonal_pilot, cfg_->GetBeamScId()
// collapses every subcarrier in a PilotScGroupSize()-wide block to the same
// index, so DoPrecode's own sequential subcarrier loop (Launch(), this
// thread only) sees several consecutive calls read the exact same
// dl_beam_matrices_ cell -- i.e. the same `mat_precoder`, hence the same Q,
// chol, LLL reduction (steps 1-3), bit for bit. Caching the last result as
// plain members (vpp_lll_cache_*, keyed on frame_id+beam_sc_id) is safe with
// no atomics: this function only ever runs on the one thread that owns this
// DoPrecode instance, and the subcarriers sharing a beam_sc_id are visited
// one after another within (or across) that same thread's Launch() calls,
// never concurrently. With freq_orthogonal_pilot off, beam_sc_id==sc_id
// always differs between calls, so the cache simply never hits (no cost
// beyond one integer comparison).
void DoPrecode::ApplyVppPerturbation(size_t frame_id, size_t beam_sc_id,
                                      const arma::cx_fmat& mat_precoder,
                                      arma::cx_fmat& mat_data, size_t ue_num,
                                      double tau, int window) {
  const size_t bs_ant_num = cfg_->BsAntNum();
  const int num_int = static_cast<int>(2 * ue_num);

  arma::fmat f_real(2 * bs_ant_num, 2 * ue_num);
  f_real.submat(0, 0, bs_ant_num - 1, ue_num - 1) = arma::real(mat_precoder);
  f_real.submat(0, ue_num, bs_ant_num - 1, 2 * ue_num - 1) =
      -arma::imag(mat_precoder);
  f_real.submat(bs_ant_num, 0, 2 * bs_ant_num - 1, ue_num - 1) =
      arma::imag(mat_precoder);
  f_real.submat(bs_ant_num, ue_num, 2 * bs_ant_num - 1, 2 * ue_num - 1) =
      arma::real(mat_precoder);

  arma::vec y(2 * ue_num);
  for (size_t i = 0; i < ue_num; i++) {
    y(i) = mat_data(i, 0).real();
    y(ue_num + i) = mat_data(i, 0).imag();
  }

  // g/b need the data vector y and are always recomputed (cheap relative to
  // the chol+LLL below); q's LLL reduction does not depend on y and is what
  // the cache below skips on a hit.
  const arma::mat f_dbl = arma::conv_to<arma::mat>::from(f_real);
  const arma::mat g = f_dbl.t() * f_dbl;
  const arma::vec b = (2.0 * tau) * (g * y);

  arma::mat q_red;
  arma::mat tmap;
  arma::mat rr;  // chol(q_red, "upper"); only meaningful when reduced.
  double sc;
  bool reduced;

  const bool cache_hit =
      cfg_->FreqOrthogonalPilot() && vpp_lll_cache_valid_ &&
      (vpp_lll_cache_frame_id_ == frame_id) &&
      (vpp_lll_cache_beam_sc_id_ == beam_sc_id) &&
      (vpp_lll_cache_num_int_ == static_cast<size_t>(num_int));

  if (cache_hit) {
    q_red = vpp_lll_cache_q_red_;
    tmap = vpp_lll_cache_tmap_;
    sc = vpp_lll_cache_sc_;
    reduced = vpp_lll_cache_reduced_;
    rr = vpp_lll_cache_rr_;
  } else {
    arma::mat q = (tau * tau) * g;
    q = 0.5 * (q + q.t());
    reduced = true;
    try {
      const arma::mat basis = arma::chol(q, "upper");
      if (cfg_->VppDisableLll()) {
        // Ablation: keep the original (unreduced) Gram coordinates. Babai
        // centering below still runs on this basis's own Cholesky factor
        // unless VppDisableBabaiStart() also forces a cold (w=0) start.
        tmap = arma::mat(num_int, num_int, arma::fill::eye);
        q_red = q;
      } else {
        tmap = LllReduce(basis, cfg_->VppLllDelta());
        q_red = tmap.t() * q * tmap;
        q_red = 0.5 * (q_red + q_red.t());
      }
    } catch (std::runtime_error&) {
      // Gram matrix too ill-conditioned to Cholesky-factor (e.g.
      // near-singular channel): fall back to searching the unreduced
      // problem around the origin, same as plain VPP, rather than crashing
      // this worker thread.
      AGORA_LOG_WARN(
          "Vpp: LLL/Babai preconditioning failed, falling back to unreduced "
          "search\n");
      q_red = q;
      tmap = arma::mat(num_int, num_int, arma::fill::eye);
      reduced = false;
    }
    sc = arma::abs(q_red).max();
    if (!(sc > 0.0) || !std::isfinite(sc)) {
      sc = 1.0;
    }
    // Also channel-only (independent of b_red): cached below alongside
    // q_red/tmap/sc so the Babai centering step for every other subcarrier
    // in this beam_sc_id group reuses it via triangular solves instead of
    // re-factoring q_red on each one.
    rr = reduced ? arma::chol(q_red, "upper") : arma::mat();

    if (cfg_->FreqOrthogonalPilot()) {
      vpp_lll_cache_valid_ = true;
      vpp_lll_cache_frame_id_ = frame_id;
      vpp_lll_cache_beam_sc_id_ = beam_sc_id;
      vpp_lll_cache_num_int_ = static_cast<size_t>(num_int);
      vpp_lll_cache_q_red_ = q_red;
      vpp_lll_cache_tmap_ = tmap;
      vpp_lll_cache_sc_ = sc;
      vpp_lll_cache_reduced_ = reduced;
      vpp_lll_cache_rr_ = rr;
    }
  }

  // Data-dependent from here on (needs b_red = tmap'*b); always recomputed.
  const arma::vec b_red = tmap.t() * b;
  arma::vec z_ctr(num_int, arma::fill::zeros);
  arma::vec b_w;
  // Smallest single-layer margin (cost gap between Babai's rounding and the
  // next-nearest integer at that layer, in q_red-native units) across all
  // reduced coordinates -- a cheap byproduct of the back-substitution below.
  // For continuous center = k + frac (k = round(center), |frac| <= 0.5),
  // that gap is rr_ii^2*(1-2|frac|): near rr_ii^2 when frac ~ 0 (k clearly
  // best), near 0 when frac ~ 0.5 (k and its neighbor are nearly tied). A
  // large minimum across layers means every coordinate rounded with a
  // comfortable margin over its runner-up, i.e. the Babai center is
  // probably already optimal; small means some coordinate is near a tie,
  // i.e. worth searching. See VppBabaiSkipMargin().
  double min_layer_gap = std::numeric_limits<double>::infinity();
  if (reduced && !cfg_->VppDisableBabaiStart()) {
    // Babai nearest-plane center via back-substitution rounding on
    // chol(Q_red) (rr, cached). rr.t()*rr == q_red, so solving
    // rr.t()*yr == -0.5*b_red directly gives the same yr that the old code
    // got by first solving the full system for z_star and then forming
    // rr*z_star -- one triangular solve instead of a generic solve plus a
    // matrix-vector product, and no fresh factorization on a cache hit.
    const arma::vec yr = arma::solve(arma::trimatl(rr.t()), -0.5 * b_red);
    for (int i = num_int - 1; i >= 0; --i) {
      double s = yr(i);
      for (int j = i + 1; j < num_int; ++j) {
        s -= rr(i, j) * z_ctr(j);
      }
      z_ctr(i) = std::round(s / rr(i, i));
      const double resid = s - rr(i, i) * z_ctr(i);  // rr(i,i) * frac
      const double gap =
          rr(i, i) * rr(i, i) - 2.0 * rr(i, i) * std::fabs(resid);
      min_layer_gap = std::min(min_layer_gap, gap);
    }
    b_w = 2.0 * (q_red * z_ctr) + b_red;
  } else {
    // Cold (w=0) start: either the Cholesky-failure fallback (tmap == I, so
    // b_red == b), or VppDisableBabaiStart() forcing a zero start in
    // whichever coordinates (LLL-reduced or not) were selected above.
    b_w = b_red;
  }

  const arma::mat q_use = q_red / sc;
  const arma::vec b_use = b_w / sc;

  // w=0 is the Babai center itself (or the origin, on the fallback path) --
  // the best available warm start either way.
  const arma::ivec init_kernel(num_int, arma::fill::zeros);
  const double babai_skip_margin = cfg_->VppBabaiSkipMargin();
  // sc normalizes q_red/b_w down to q_use/b_use (the units the configured
  // threshold and the solvers' own "energy" are in), so the gap has to be
  // normalized the same way to compare on equal footing.
  const bool skip_search =
      reduced && (babai_skip_margin > 0.0) &&
      (min_layer_gap / sc >= babai_skip_margin * babai_skip_margin);
  VppSolverResult result;
  if (skip_search) {
    result.solution.zeros(num_int);
    result.energy = 0.0;
  } else {
    switch (cfg_->BeamformingAlgo()) {
      case CommsLib::BeamformingAlgorithm::kSE:
        result = VppSolveSphere(q_use, b_use, init_kernel,
                                VppSphereConfig{-window, window});
        break;
      case CommsLib::BeamformingAlgorithm::kFCSE:
        result = VppSolveFixedComplexity(
            q_use, b_use, init_kernel,
            VppFixedComplexityConfig{-window, window,
                                     cfg_->VppFcseFullLayers()});
        break;
      default: {
        VppSolverConfig solver_cfg;
        solver_cfg.value_min = -window;
        solver_cfg.value_max = window;
        solver_cfg.num_replicas = cfg_->VppNumReplicas();
        solver_cfg.runs = cfg_->VppRuns();
        solver_cfg.iters = cfg_->VppIters();
        solver_cfg.neighbor_window =
            std::min(cfg_->VppNeighborWindow(), static_cast<size_t>(window));
        solver_cfg.beta_min = cfg_->VppBetaMin();
        solver_cfg.beta_max = cfg_->VppBetaMax();
        solver_cfg.early_stop_patience = cfg_->VppEarlyStopPatience();
        solver_cfg.run_patience = cfg_->VppRunPatience();
        result = VppSolveInteger(q_use, b_use, init_kernel, solver_cfg);
      }
    }
  }

  const arma::vec w = arma::conv_to<arma::vec>::from(result.solution);
  const arma::vec z = reduced ? (z_ctr + w) : w;
  arma::vec l = tmap * z;

  // Never transmit at higher energy than plain ZF (l=0): mirrors a
  // colleague's hardware VPP design (paramax_ORBIT vp_solver.h), whose
  // exact solvers are seeded from the unperturbed vector so the chosen
  // perturbation can only tie or beat ZF's own transmit energy, never
  // lose to it. LLL/Babai/the search all target the energy minimum but
  // offer no such guarantee on their own (the search window is centered
  // on Babai's point, not on l=0).
  {
    const arma::vec y_pert = y + tau * l;
    const double e_pert = arma::as_scalar(y_pert.t() * g * y_pert);
    const double e_zero = arma::as_scalar(y.t() * g * y);
    if (e_pert > e_zero) {
      l.zeros();
    }
  }

  for (size_t i = 0; i < ue_num; i++) {
    const auto perturb_re = static_cast<float>(tau * l(i));
    const auto perturb_im = static_cast<float>(tau * l(ue_num + i));
    mat_data(i, 0) += arma::cx_float(perturb_re, perturb_im);
  }
}
