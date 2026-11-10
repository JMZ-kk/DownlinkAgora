// Copyright (c) 2018-2020, Rice University
// RENEW OPEN SOURCE LICENSE: http://renew-wireless.org/license

/**
 * @file config.h
 * @brief Declaration file for the configuration class which importants
 * json configuration values into class variables
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#include <atomic>
#include <cstddef>
#include <string>
#include <vector>

#include "armadillo"
#include "common_typedef_sdk.h"
#include "comms-lib.h"
#include "concurrent_queue_wrapper.h"
#include "framestats.h"
#include "gettime.h"
#include "mac_utils.h"
#include "memory_manage.h"
#include "nlohmann/json.hpp"
#include "symbols.h"
#include "utils.h"

class Config {
 public:
  static constexpr bool kDebugRecipCal = false;
  // Constructor
  explicit Config(std::string jsonfilename);
  ~Config();

  inline void Running(bool value) { this->running_.store(value); }
  inline bool Running() const { return this->running_.load(); }
  inline size_t BsAntNum() const { return this->bs_ant_num_; }
  inline void BsAntNum(size_t n_bs_ant) { this->bs_ant_num_ = n_bs_ant; }

  // Inline accessors (basic types)
  inline size_t BfAntNum() const { return this->bf_ant_num_; }
  inline size_t UeNum() const { return this->ue_num_; }
  inline size_t UeAntNum() const { return this->ue_ant_num_; }
  inline bool AdaptUes() const { return this->adapt_ues_; }
  inline size_t UeAntOffset() const { return this->ue_ant_offset_; }
  inline size_t UeAntTotal() const { return this->ue_ant_total_; }

  inline size_t OfdmCaNum() const { return this->ofdm_ca_num_; }
  inline size_t CpLen() const { return this->cp_len_; }
  inline size_t OfdmDataNum() const { return this->ofdm_data_num_; }
  inline size_t OfdmDataStart() const { return this->ofdm_data_start_; }

  inline size_t OfdmDataStop() const { return this->ofdm_data_stop_; }
  inline size_t OfdmPilotSpacing() const { return this->ofdm_pilot_spacing_; }

  inline bool HwFramer() const { return this->hw_framer_; }
  inline bool UeHwFramer() const { return this->ue_hw_framer_; }
  inline size_t UeResyncPeriod() const { return this->ue_resync_period_; }
  inline double FreqGhz() const { return this->freq_ghz_; };
  inline double Freq() const { return this->freq_; }
  inline double Rate() const { return this->rate_; }
  inline double Nco() const { return this->nco_; }

  inline double RadioRfFreq() const { return this->radio_rf_freq_; }
  inline double BwFilter() const { return this->bw_filter_; }
  inline bool SingleGain() const { return this->single_gain_; }
  inline double TxGainA() const { return this->tx_gain_a_; }
  inline double RxGainA() const { return this->rx_gain_a_; }
  inline double TxGainB() const { return this->tx_gain_b_; }
  inline double RxGainB() const { return this->rx_gain_b_; }
  inline double CalibTxGainA() const { return this->calib_tx_gain_a_; }
  inline double CalibTxGainB() const { return this->calib_tx_gain_b_; }
  inline double ClientTxGainA(size_t id) const {
    return this->client_tx_gain_a_.at(id);
  }
  inline double ClientRxGainA(size_t id) const {
    return this->client_rx_gain_a_.at(id);
  }
  inline double ClientTxGainB(size_t id) const {
    return this->client_tx_gain_b_.at(id);
  }
  inline double ClientRxGainB(size_t id) const {
    return this->client_rx_gain_b_.at(id);
  }
  inline const std::vector<double>& ClientTxGainA() const {
    return this->client_tx_gain_a_;
  }
  inline const std::vector<double>& ClientRxGainA() const {
    return this->client_rx_gain_a_;
  }
  inline const std::vector<double>& ClientTxGainB() const {
    return this->client_tx_gain_b_;
  }
  inline const std::vector<double>& ClientRxGainB() const {
    return this->client_rx_gain_b_;
  }
  inline size_t NumCells() const { return this->num_cells_; }
  inline size_t NumRadios() const { return this->num_radios_; }
  inline size_t InitCalibRepeat() const { return this->init_calib_repeat_; }

  inline size_t NumChannels() const { return this->num_channels_; }
  inline size_t NumUeChannels() const { return this->num_ue_channels_; }
  inline size_t RefAnt(size_t id) const { return this->ref_ant_.at(id); }
  inline size_t RefRadio(size_t id) const { return this->ref_radio_.at(id); }
  inline size_t BeaconAnt() const { return this->beacon_ant_; }
  inline size_t BeaconLen() const { return this->beacon_len_; }

  inline bool DynamicCoreAlloc() const {
    return this->dynamic_core_allocation_;
  }
  inline bool SmoothCalib() const { return this->smooth_calib_; }
  inline bool Beamsweep() const { return this->beamsweep_; }
  inline bool SampleCalEn() const { return this->sample_cal_en_; }
  inline bool ImbalanceCalEn() const { return this->imbalance_cal_en_; }
  inline size_t BeamformingAlgo() const { return this->beamforming_algo_; }
  inline std::string Beamforming() const { return this->beamforming_str_; }
  inline size_t SpatialStreamsNum() const { return this->num_spatial_streams_; }
  // VPP (Vector Perturbation Precoding) integer-QP solver knobs: N_R replicas,
  // N_s runs, N_C sweeps, proposal half-width, and the (beta_min, beta_max)
  // geometric ladder for Q,b normalized by max|Q|.
  inline size_t VppNumReplicas() const { return this->vpp_num_replicas_; }
  inline size_t VppRuns() const { return this->vpp_runs_; }
  inline size_t VppIters() const { return this->vpp_iters_; }
  inline size_t VppNeighborWindow() const {
    return this->vpp_neighbor_window_;
  }
  inline double VppBetaMin() const { return this->vpp_beta_min_; }
  inline double VppBetaMax() const { return this->vpp_beta_max_; }
  // Vpp-only: LLL Lovász parameter and the search window half-width used
  // around the Babai center in reduced coordinates (<= 0: paper's bound + 2).
  inline double VppLllDelta() const { return this->vpp_lll_delta_; }
  inline int VppLllWindow() const { return this->vpp_lll_window_; }
  inline size_t VppFcseFullLayers() const {
    return this->vpp_fcse_full_layers_;
  }
  // Vpp: rescale each DL data symbol so it carries the same total energy
  // plain ZF would have sent (BS: doifft.cc), and let the UE recover that gain
  // from the DMRS tones (ue_worker.cc). Without it Vpp saves energy nobody
  // spends and only pays the modulo loss.
  inline bool VppPowerNorm() const { return this->vpp_power_norm_; }
  inline bool VppPeriodicLlr() const { return this->vpp_periodic_llr_; }
  // Vpp: skip the replicas/runs/iters search entirely and use the Babai
  // center as-is when every reduced-coordinate layer's single-flip cost
  // (in the normalized q_use/b_use energy units the solver itself reports)
  // is at least this margin -- i.e. Babai looks confidently optimal, so the
  // search is unlikely to find anything better. 0 (default) = always search
  // (no behavior change). See doprecode.cc::ApplyVppPerturbation.
  inline double VppBabaiSkipMargin() const {
    return this->vpp_babai_skip_margin_;
  }
  // Vpp: a fixed, data-independent amplitude boost applied to every
  // perturbed DL data symbol in doifft.cc, in place of the per-symbol
  // sqrt(sum e_ref/sum e_vpp) exact energy match (vpp_power_stats.h). 0
  // (default) = current behavior (exact per-symbol match). > 0 = one
  // common scale across every symbol, matching a fixed hardware VPP
  // design the team has compared against: the UE's DMRS-based inv_gain
  // (ue_worker.cc) still measures whatever gain is actually present (it
  // does not need to know this value), so this is safe to change without
  // touching the receiver. The per-symbol exact match recovers each
  // symbol's own energy savings precisely but leaves a residual that Vpp's
  // modulo-folded decision regions amplify (see
  // project-vpp-full-load-findings memory); a fixed constant has no such
  // residual under noiseless pilots, at the cost of not spending 100% of
  // each symbol's actual savings.
  inline double VppFixedGain() const { return this->vpp_fixed_gain_; }
  // Burst-level peak-amplitude AGC (see doifft.cc), applied identically to
  // every beamforming algorithm (not Vpp-specific) in place of
  // VppPowerNorm()/VppFixedGain(): one shared gain, re-derived from a
  // running estimate of the actually-realized time-domain peak sample
  // magnitude (decayed max over recent DL symbols, all antennas), targeting
  // a fixed fraction of full scale. Mirrors a colleague's hardware VPP
  // design (paramax_ORBIT ofdm_modulate.h): peak_payload is scanned once
  // per burst over every antenna and symbol, then applied_gain =
  // 0.7/peak_payload is shared by all of them -- the burst's own realized
  // peak, not a per-symbol energy match or a hand-picked constant. Agora
  // streams symbols in real time and can't see a frame's peak before
  // transmitting it, so this uses the running estimate from recently-sent
  // symbols instead of the true in-burst peak; everything else is the same
  // idea. False (default) = current behavior.
  inline bool TxPeakAgc() const { return this->tx_peak_agc_; }
  // A plain, algorithm-agnostic constant amplitude multiplier applied to
  // every DL symbol/antenna's IFFT input, for ANY beamforming algorithm
  // (unlike VppFixedGain(), which only applies under
  // IsPerturbationPrecoder()). 0 (default, i.e. 1.0x) = current behavior.
  // Exists to directly test whether a given algorithm's own transmitted
  // signal (e.g. plain ZF's P*u) has unused int16 headroom below the
  // clipping ceiling that a fixed multiplier can safely spend for more
  // SNR, independent of any Vpp-specific mechanism.
  inline double TxFixedGain() const { return this->tx_fixed_gain_; }
  // Path to a CSV file of "frame,dl_symbol_idx,gain" rows (see doifft.cc):
  // the exact per-symbol gain c* = target/true_peak, measured from an
  // earlier gain=1 run of the SAME data_generator output (same payload
  // bits, not regenerated) so it is the true, non-lagged optimum for this
  // specific transmission, not an online estimate. Empty (default) =
  // disabled. Highest priority: active whenever non-empty, regardless of
  // TxFixedGain/TxPeakAgc/VppPowerNorm.
  inline const std::string& TxExactGainFile() const {
    return this->tx_exact_gain_file_;
  }
  // A fixed, per-subcarrier transmit-energy target, independent of P and u
  // (see doprecode.cc's PrecodedEnergy/track_power_ and doifft.cc): every
  // algorithm's own actual precoded energy (||P*u||^2 for Zf, ||P*(u+tau*l)||^2
  // for Vpp) is rescaled, per data subcarrier, to exactly this constant --
  // not matched against the OTHER scheme's energy (VppPowerNorm's job) and
  // not a peak/clipping bound (TxPeakAgc/TxExactGainFile's job). With this
  // on, transmit power no longer depends on the random data or which
  // algorithm is active, so SNR = TxEnergyTarget() / NoiseLevel()^2 up to
  // the channel's own (H-dependent) gain -- the same idea as
  // 7_8_26_intmaxbm_downlink.cpp's "symbol" receiver (Pt = ||x_t||^2,
  // g = 1/sqrt(Pt)), generalized to rescale TO an arbitrary target instead
  // of always unit energy, and applied in Agora's own pipeline instead of
  // an offline analysis script. 0 (default) = disabled.
  inline double TxEnergyTarget() const { return this->tx_energy_target_; }
  // Ablation knobs (both default false = current behavior). Together they
  // give the same 2x2 {LLL, no-LLL} x {Babai start, cold/zero start} design
  // as the PreQ paper's Figures 3-4: does the search's own coordinates
  // (LLL) matter, independent of where it starts (Babai vs zero)? See
  // doprecode.cc::ApplyVppPerturbation.
  // true: skip LLL -- search directly in the original (unreduced) Gram
  // coordinates instead of the LLL-reduced basis. Babai centering (if not
  // also disabled below) still runs, just on the unreduced basis's own
  // Cholesky factor.
  inline bool VppDisableLll() const { return this->vpp_disable_lll_; }
  // true: skip the Babai back-substitution rounding and start the search at
  // the origin (w=0) regardless of whether LLL ran -- "cold start" in
  // whichever coordinates VppDisableLll() selects.
  inline bool VppDisableBabaiStart() const {
    return this->vpp_disable_babai_start_;
  }
  // Vpp (VppSolveInteger only, i.e. the default/"VPP" solver, not SE/FCSE):
  // bail out of a run's sweep loop early once its best-so-far energy has
  // gone this many consecutive sweeps without improving. 0 (default) = never
  // early-stop, always run all vpp_iters sweeps (no behavior change). See
  // VppSolverConfig::early_stop_patience / VppSolveInteger.
  inline size_t VppEarlyStopPatience() const {
    return this->vpp_early_stop_patience_;
  }
  // Vpp (VppSolveInteger only): stop issuing further restarts (vpp_runs)
  // once this many consecutive restarts in a row failed to beat the
  // best energy found so far. 0 (default) = never (no behavior change). See
  // VppSolverConfig::run_patience / VppSolveInteger.
  inline size_t VppRunPatience() const { return this->vpp_run_patience_; }
  inline bool ExternalRefNode(size_t id) const {
    return this->external_ref_node_.at(id);
  }
  inline std::string Channel() const { return this->channel_; }
  inline std::string UeChannel() const { return this->ue_channel_; }

  // Groups for Downlink Recip Cal
  // Returns antenna number for rec cal dl symbol
  // Assumes that there are the same number of dl cal symbols in each frame
  inline size_t RecipCalDlAnt(size_t frame_id, size_t dl_cal_symbol) const {
    assert(this->Frame().GetSymbolType(dl_cal_symbol) == SymbolType::kCalDL);
    const size_t dl_cal_offset = (frame_id * frame_.NumDLCalSyms()) +
                                 frame_.GetDLCalSymbolIdx(dl_cal_symbol);

    const size_t tx_ant = dl_cal_offset % bf_ant_num_;

    if (kDebugRecipCal) {
      std::printf("RecipCalDlAnt (Frame %zu, Symbol %zu) tx antenna %zu\n",
                  frame_id, dl_cal_symbol, tx_ant);
    }
    return (tx_ant);
  };

  inline size_t ModifyRecCalIndex(size_t previous_index,
                                  int mod_value = 0) const {
    return (previous_index + mod_value) % kFrameWnd;
  }

  inline size_t RecipCalIndex(size_t frame_id) const {
    const size_t frame_cal_idx = frame_id / RecipCalFrameCnt();
    return ModifyRecCalIndex(frame_cal_idx);
  }

  // Returns the cal index if ant tx dl cal pilots this frame
  // SIZE_MAX otherwise
  inline size_t RecipCalUlRxIndex(size_t frame_id, size_t ant) const {
    const size_t num_frames_for_full_cal = RecipCalFrameCnt();
    const size_t num_cal_per_idx = frame_.NumDLCalSyms();
    const size_t cal_offset = (frame_id % num_frames_for_full_cal);
    const size_t tx_ant_start = cal_offset * num_cal_per_idx;
    const size_t tx_ant_end = tx_ant_start + (num_cal_per_idx - 1);

    size_t cal_ind;
    if ((ant >= tx_ant_start) && (ant <= tx_ant_end)) {
      cal_ind = RecipCalIndex(frame_id);
    } else {
      cal_ind = SIZE_MAX;
    }

    if (kDebugRecipCal) {
      std::printf(
          "RecipCalUlRxIndex (Frame %zu, Antenna %zu) index %zu - Start %zu, "
          "End %zu, full %zu\n",
          frame_id, ant, cal_ind, tx_ant_start, tx_ant_end,
          num_frames_for_full_cal);
    }
    return (cal_ind);
  };

  // Returns the number of frames to obtain a full set of RecipCal data
  // assumes that bf_ant_num_ % frame_.NumDLCalSyms() == 0
  inline size_t RecipCalFrameCnt() const {
    if ((frame_.IsRecCalEnabled() == false) || (frame_.NumDLCalSyms() == 0)) {
      return 0;
    } else {
      assert((bf_ant_num_ % frame_.NumDLCalSyms()) == 0);
      return bf_ant_num_ / frame_.NumDLCalSyms();
    }
  }

  inline std::vector<size_t> ExcludedCores() const { return this->excluded_; }
  inline size_t CoreOffset() const { return this->core_offset_; }
  inline size_t WorkerThreadNum() const { return this->worker_thread_num_; }
  inline size_t SocketThreadNum() const { return this->socket_thread_num_; }
  inline size_t UeCoreOffset() const { return this->ue_core_offset_; }
  inline size_t UeWorkerThreadNum() const {
    return this->ue_worker_thread_num_;
  }
  inline size_t UeSocketThreadNum() const {
    return this->ue_socket_thread_num_;
  }

  inline size_t FftThreadNum() const { return this->fft_thread_num_; }
  inline size_t DemulThreadNum() const { return this->demul_thread_num_; }
  inline size_t DecodeThreadNum() const { return this->decode_thread_num_; }
  inline size_t BeamThreadNum() const { return this->beam_thread_num_; }
  inline size_t DemulBlockSize() const { return this->demul_block_size_; }

  inline size_t DemulEventsPerSymbol() const {
    return this->demul_events_per_symbol_;
  }
  inline size_t BeamBlockSize() const { return this->beam_block_size_; }
  inline size_t BeamEventsPerSymbol() const {
    return this->beam_events_per_symbol_;
  }
  inline size_t FftBlockSize() const { return this->fft_block_size_; }

  inline size_t EncodeBlockSize() const { return this->encode_block_size_; }
  inline bool FreqOrthogonalPilot() const {
    return this->freq_orthogonal_pilot_;
  }
  inline size_t PilotScGroupSize() const { return this->pilot_sc_group_size_; }
  inline size_t OfdmTxZeroPrefix() const { return this->ofdm_tx_zero_prefix_; }
  inline size_t OfdmTxZeroPostfix() const {
    return this->ofdm_tx_zero_postfix_;
  }
  inline size_t OfdmRxZeroPrefixBs() const {
    return this->ofdm_rx_zero_prefix_bs_;
  }

  inline size_t OfdmRxZeroPrefixCalUl() const {
    return this->ofdm_rx_zero_prefix_cal_ul_;
  }
  void OfdmRxZeroPrefixCalUl(size_t prefix) {
    this->ofdm_rx_zero_prefix_cal_ul_ = prefix;
  }
  inline size_t OfdmRxZeroPrefixCalDl() const {
    return this->ofdm_rx_zero_prefix_cal_dl_;
  }
  void OfdmRxZeroPrefixCalDl(const size_t prefix) {
    this->ofdm_rx_zero_prefix_cal_dl_ = prefix;
  }
  inline size_t OfdmRxZeroPrefixClient() const {
    return this->ofdm_rx_zero_prefix_client_;
  }
  inline size_t SampsPerSymbol() const { return this->samps_per_symbol_; }
  inline size_t SampsPerFrame() const {
    return this->frame_.NumTotalSyms() * this->samps_per_symbol_;
  }
  inline size_t PacketLength() const { return this->packet_length_; }

  inline float Scale() const { return this->scale_; }
  inline bool BigstationMode() const { return this->bigstation_mode_; }
  inline size_t DlPacketLength() const { return this->dl_packet_length_; }
  inline bool ScrambleEnabled() const { return this->scramble_enabled_; }
  inline const nlohmann::json& DefaultMCSParams(Direction dir) const {
    return dir == Direction::kUplink ? this->ul_mcs_params_
                                     : this->dl_mcs_params_;
  }
  inline const MacUtils& MacParams() const { return this->mac_params_; }

  inline std::string UeServerAddr() const { return this->ue_server_addr_; }
  inline std::string BsServerAddr() const { return this->bs_server_addr_; }

  inline std::string UeRruAddr() const { return this->ue_rru_addr_; }
  inline std::string BsRruAddr() const { return this->bs_rru_addr_; }

  inline int BsServerPort() const { return this->bs_server_port_; }
  inline int BsRruPort() const { return this->bs_rru_port_; }
  inline int UeServerPort() const { return this->ue_server_port_; }
  inline int UeRruPort() const { return this->ue_rru_port_; }

  inline size_t FramesToTest() const { return this->frames_to_test_; }
  inline size_t FrameToProfile() const { return this->frame_to_profile_; }
  inline float NoiseLevel() const { return this->noise_level_; }
  // data_generator only: how the per-UE power/noise normalization in the
  // synthetic UL RX generation is applied (see data_generator.cc). 0 =
  // unnormalized (noise std is the raw noise_level); 1 = divide the signal
  // by sqrt(UeAntNum()) to hold total transmit power constant across UE
  // counts; 2 = equivalently scale the noise up by sqrt(UeAntNum()) instead
  // of scaling the signal down. Either nonzero mode additionally divides
  // the first (pilot) symbol's noise by sqrt(PilotScGroupSize()).
  inline int NoiseMode() const { return this->noise_mode_; }
  // data_generator only: number of independent (bits, noise, channel H)
  // realizations written to the UL test-vector files, cycled frame-major as
  // [data_frame][sched][...]. 1 reproduces the legacy single-realization
  // layout byte-for-byte. Downlink test vectors always use data_frame 0 only.
  inline size_t DistinctDataFrames() const {
    return this->distinct_data_frames_;
  }
  // chsim only: leave the UL pilot symbols noise-free ("perfect BS-side CSI").
  inline bool NoiselessPilots() const { return this->noiseless_pilots_; }
  // chsim only: leave the DL pilot symbols noise-free ("perfect UE-side CSI").
  // Defaults to NoiselessPilots() so an unset config keeps the old combined
  // on/off behavior; set explicitly to decouple UL-CSI error from DL-CSI
  // error (e.g. to isolate which one drives a BER regression).
  inline bool DlNoiselessPilots() const { return this->dl_noiseless_pilots_; }
  // chsim only: nonzero seeds the Rayleigh channel-matrix sequence so repeated
  // runs see identical channels (0 = unseeded, the default).
  inline uint64_t ChannelSeed() const { return this->channel_seed_; }

  inline bool FreqDomainChannel() const { return this->freq_domain_channel_; }
  inline uint16_t DpdkNumPorts() const { return this->dpdk_num_ports_; }
  inline uint16_t DpdkPortOffset() const { return this->dpdk_port_offset_; }

  inline std::string SchedulerType() const { return this->scheduler_type_; }

  inline const std::string& DpdkMacAddrs() const {
    return this->dpdk_mac_addrs_;
  }

  inline size_t BsMacRxPort() const { return this->bs_mac_rx_port_; }
  inline size_t BsMacTxPort() const { return this->bs_mac_tx_port_; }

  inline size_t UeMacRxPort() const { return this->ue_mac_rx_port_; }
  inline size_t UeMacTxPort() const { return this->ue_mac_tx_port_; }

  inline std::string UeAppRxAddr() const { return this->ue_app_rx_addr_; }
  inline std::string BsAppRxAddr() const { return this->bs_app_rx_addr_; }
  inline size_t UeAppRxPort() const { return this->ue_app_rx_port_; }
  inline size_t BsAppRxPort() const { return this->bs_app_rx_port_; }

  inline std::string RpRemoteHostName() const {
    return this->rp_remote_host_name_;
  }
  inline size_t RpRxPort() const { return this->rp_rx_port_; }
  inline size_t RpTxPort() const { return this->rp_tx_port_; }

  inline const std::string& LogListenerAddr() const {
    return this->log_listener_addr_;
  }

  inline size_t LogListenerPort() const { return this->log_listener_port_; }

  inline size_t LogScNum() const { return this->log_sc_num_; }
  inline bool LogTimestamp() const { return this->log_timestamp_; }

  /* Inline accessors (complex types) */
  inline const std::vector<int>& ClTxAdvance() const {
    return this->cl_tx_advance_;
  }
  inline const std::vector<float>& ClCorrScale() const {
    return this->cl_corr_scale_;
  }

  inline const FrameStats& Frame() const { return this->frame_; }
  inline const std::vector<std::complex<float>>& PilotCf32() const {
    return this->pilot_cf32_;
  };
  inline const std::vector<std::complex<float>>& GoldCf32() const {
    return this->gold_cf32_;
  };
  inline const std::vector<uint32_t>& Coeffs() const { return this->coeffs_; };

  inline const std::vector<uint32_t>& Pilot() const { return this->pilot_; };
  inline const std::vector<uint32_t>& Beacon() const { return this->beacon_; };
  inline const complex_float* Pilots() const { return this->pilots_; };
  inline const complex_float* PilotsSgn() const { return this->pilots_sgn_; };
  inline const std::vector<std::complex<float>>& CommonPilot() const {
    return this->common_pilot_;
  };
  inline const std::vector<std::string>& RadioId() const {
    return this->radio_id_;
  };
  inline const std::vector<std::string>& HubId() const {
    return this->hub_id_;
  };
  inline const std::vector<std::string>& UeRadioId() const {
    return this->ue_radio_id_;
  };
  inline const std::vector<std::string>& UeRadioName() const {
    return this->ue_radio_name_;
  };
  inline const std::vector<size_t>& CellId() const { return this->cell_id_; }

  // non-const (can modify)
  inline Table<complex_float>& UeSpecificPilot() {
    return this->ue_specific_pilot_;
  };
  inline Table<std::complex<int16_t>>& UeSpecificPilotT() {
    return this->ue_specific_pilot_t_;
  };
  inline std::vector<std::complex<int16_t>>& PilotCi16() {
    return this->pilot_ci16_;
  };
  inline std::vector<std::complex<int16_t>>& PilotUeCi16(size_t ue_id,
                                                         size_t pilot_idx) {
    return this->pilot_ue_ci16_.at(ue_id).at(pilot_idx);
  };
  inline const arma::uvec& PilotUeSc(size_t ue_id) const {
    return this->pilot_ue_sc_.at(ue_id);
  };
  inline std::vector<std::complex<int16_t>>& BeaconCi16() {
    return this->beacon_ci16_;
  };

  inline Table<int8_t>& DlModBits() { return this->dl_mod_bits_; }
  inline Table<int8_t>& UlModBits() { return this->ul_mod_bits_; }
  inline Table<complex_float>& UlIqF() { return this->ul_iq_f_; }
  inline Table<complex_float>& DlIqF() { return this->dl_iq_f_; }
  inline Table<std::complex<int16_t>>& UlIqT() { return this->ul_iq_t_; }
  inline Table<std::complex<int16_t>>& DlIqT() { return this->dl_iq_t_; }

  // Public functions
  void GenPilots();
  void LoadUplinkData();
  void LoadDownlinkData();
  void LoadTestVectors();

  /// Return total number of data symbols of all frames in a buffer
  /// that holds data of kFrameWnd frames
  inline size_t GetTotalDataSymbolIdx(size_t frame_id, size_t symbol_id) const {
    return ((frame_id % kFrameWnd) * this->frame_.NumDataSyms() + symbol_id);
  }

  /// Return total number of uplink data symbols of all frames in a buffer
  /// that holds data of kFrameWnd frames
  inline size_t GetTotalDataSymbolIdxUl(size_t frame_id,
                                        size_t data_symbol_idx_ul) const {
    return ((frame_id % kFrameWnd) * this->frame_.NumUlDataSyms() +
            data_symbol_idx_ul);
  }

  /// Return total number of uplink data symbols of all frames in a buffer
  /// that holds data of kFrameWnd frames
  inline size_t GetTotalSymbolIdxUl(size_t frame_id,
                                    size_t symbol_idx_ul) const {
    return ((frame_id % kFrameWnd) * this->frame_.NumULSyms() + symbol_idx_ul);
  }

  /// Return total number of downlink data symbols of all frames in a buffer
  /// that holds data of kFrameWnd frames
  inline size_t GetTotalDataSymbolIdxDl(size_t frame_id,
                                        size_t data_symbol_idx_dl) const {
    return ((frame_id % kFrameWnd) * this->frame_.NumDlDataSyms() +
            data_symbol_idx_dl);
  }

  /// Return total number of downlink data and pilot symbols of all frames in a buffer
  /// that holds data of kFrameWnd frames
  inline size_t GetTotalSymbolIdxDl(size_t frame_id,
                                    size_t symbol_idx_dl) const {
    return ((frame_id % kFrameWnd) * this->frame_.NumDLSyms() + symbol_idx_dl);
  }

  /// Return total number of downlink data, pilot, and broadcast symbols of all frames
  /// in a buffer that holds data of kFrameWnd frames
  inline size_t GetTotalSymbolIdxDlBcast(size_t frame_id, size_t symbol_id) {
    const size_t symbol_idx_dl =
        symbol_id < this->frame_.GetDLSymbol(0)
            ? this->frame_.GetDLControlSymbolIdx(symbol_id)
            : this->frame_.GetDLSymbolIdx(symbol_id) +
                  this->frame_.NumDlControlSyms();
    return (frame_id % kFrameWnd) *
               (this->frame_.NumDlControlSyms() + this->frame_.NumDLSyms()) +
           symbol_idx_dl;
  }

  /// Return the symbol duration in seconds
  inline double GetSymbolDurationSec() const {
    return (this->samps_per_symbol_ / this->rate_);
  }

  /// Return the frame duration in seconds
  inline double GetFrameDurationSec() const {
    return ((this->frame_.NumTotalSyms() * this->samps_per_symbol_) /
            this->rate_);
  }

  /// Fetch the data buffer for this frame and symbol ID. The symbol must
  /// be an uplink symbol.
  inline complex_float* GetDataBuf(Table<complex_float>& data_buffers,
                                   size_t frame_id, size_t symbol_id) const {
    size_t frame_slot = frame_id % kFrameWnd;
    size_t symbol_offset = (frame_slot * this->frame_.NumULSyms()) +
                           this->frame_.GetULSymbolIdx(symbol_id);
    return data_buffers[symbol_offset];
  }

  /// Return the subcarrier ID which we should refer to for the beamweight
  /// matrices of subcarrier [sc_id].
  inline size_t GetBeamScId(size_t sc_id) const {
    return this->freq_orthogonal_pilot_
               ? sc_id - (sc_id % this->pilot_sc_group_size_)
               : sc_id;
  }

  /// Get the calibration buffer for this frame and subcarrier ID
  inline complex_float* GetCalibBuffer(Table<complex_float>& calib_buffer,
                                       size_t frame_id, size_t sc_id) const {
    size_t frame_slot = frame_id % kFrameWnd;
    return &calib_buffer[frame_slot][sc_id * bs_ant_num_];
  }

  /// Get encoded_buffer for this frame, symbol, user and code block ID
  inline int8_t* GetModBitsBuf(Table<int8_t>& mod_bits_buffer, Direction dir,
                               size_t frame_id, size_t symbol_id, size_t ue_id,
                               size_t sc_id) const {
    size_t total_data_symbol_id;
    size_t ofdm_data_num;
    if (dir == Direction::kDownlink) {
      total_data_symbol_id = GetTotalDataSymbolIdxDl(frame_id, symbol_id);
      ofdm_data_num = GetOFDMDataNum();
    } else {
      total_data_symbol_id = GetTotalDataSymbolIdxUl(frame_id, symbol_id);
      ofdm_data_num = this->ofdm_data_num_;
    }

    return &mod_bits_buffer[total_data_symbol_id]
                           [Roundup<64>(ofdm_data_num) * ue_id + sc_id];
  }

  // Returns the number of pilot subcarriers in downlink symbols used for
  // phase tracking
  inline size_t GetOFDMPilotNum() const {
    return ofdm_data_num_ / ofdm_pilot_spacing_;
  }

  inline size_t GetOFDMDataNum() const {
    return ofdm_data_num_ - GetOFDMPilotNum();
  }

  inline size_t GetOFDMCtrlNum() const {
    return ofdm_data_num_ - 2 * GetOFDMPilotNum();
  }

  inline size_t GetOFDMDataIndex(size_t sc_id) const {
    return dl_symbol_data_id_.at(sc_id);
  }

  inline size_t GetOFDMCtrlIndex(size_t sc_id) const {
    return dl_symbol_ctrl_id_.at(sc_id);
  }

  inline bool IsDataSubcarrier(size_t sc_id) const {
    return dl_symbol_map_.at(sc_id) == SubcarrierType::kData;
  }
  inline bool IsControlSubcarrier(size_t sc_id) const {
    return control_symbol_map_.at(sc_id) == SubcarrierType::kData;
  }
  inline const std::string& ConfigFilename() const { return config_filename_; }
  inline const std::string& TraceFilename() const { return trace_file_; }
  inline const std::string& Timestamp() const { return timestamp_; }
  inline const std::vector<std::string>& UlTxFreqDataFiles() const {
    return ul_tx_f_data_files_;
  }

  inline void TryEnqueueLogStatsMaster(
      moodycamel::ConcurrentQueue<EventData>* mc_queue,
      moodycamel::ProducerToken* producer_token, const EventData& event,
      size_t frame_id, size_t symbol_id) {
    size_t enqueue_start_tsc = 0;
    size_t enqueue_end_tsc = 0;
    if (frame_id == this->frame_to_profile_) {
      enqueue_start_tsc = GetTime::WorkerRdtsc();
    }
    TryEnqueueFallback(mc_queue, producer_token, event);
    if (frame_id == this->frame_to_profile_) {
      enqueue_end_tsc = GetTime::WorkerRdtsc();
      enqueue_stats_[symbol_id][enqueue_stats_id_.at(symbol_id)].tsc_start_ =
          enqueue_start_tsc;
      enqueue_stats_[symbol_id][enqueue_stats_id_.at(symbol_id)].tsc_end_ =
          enqueue_end_tsc;
      enqueue_stats_[symbol_id][enqueue_stats_id_.at(symbol_id)].event_type_ =
          event.event_type_;
      enqueue_stats_id_.at(symbol_id)++;
    }
  }

  inline void TryEnqueueLogStatsMaster(
      moodycamel::ConcurrentQueue<EventData>* mc_queue, const EventData& event,
      size_t frame_id, size_t symbol_id) {
    size_t enqueue_start_tsc = 0;
    size_t enqueue_end_tsc = 0;
    if (frame_id == this->frame_to_profile_) {
      enqueue_start_tsc = GetTime::WorkerRdtsc();
    }
    TryEnqueueFallback(mc_queue, event);
    if (frame_id == this->frame_to_profile_) {
      enqueue_end_tsc = GetTime::WorkerRdtsc();
      enqueue_stats_[symbol_id][enqueue_stats_id_.at(symbol_id)].tsc_start_ =
          enqueue_start_tsc;
      enqueue_stats_[symbol_id][enqueue_stats_id_.at(symbol_id)].tsc_end_ =
          enqueue_end_tsc;
      enqueue_stats_[symbol_id][enqueue_stats_id_.at(symbol_id)].event_type_ =
          event.event_type_;
      enqueue_stats_id_.at(symbol_id)++;
    }
  }

  inline void UpdateDequeueTscWorker(int tid, size_t frame_id,
                                     size_t dequeue_tsc,
                                     size_t valid_dequeue_tsc) {
    total_worker_dequeue_tsc_[tid][frame_id] += dequeue_tsc;
    total_worker_valid_dequeue_tsc_[tid][frame_id] += valid_dequeue_tsc;
  }

  inline void UpdateEnqueueTscWorker(int tid, size_t frame_id,
                                     size_t enqueue_tsc) {
    total_worker_enqueue_tsc_[tid][frame_id] += enqueue_tsc;
    worker_num_valid_enqueue_[tid][frame_id]++;
  }

  inline void LogEnqueueStatsWorker(int tid, size_t symbol_id, size_t start_tsc,
                                    size_t end_tsc, EventType event_type) {
    size_t id = worker_enqueue_stats_id_[tid][symbol_id];
    worker_enqueue_stats_[tid][symbol_id][id].tsc_start_ = start_tsc;
    worker_enqueue_stats_[tid][symbol_id][id].tsc_end_ = end_tsc;
    worker_enqueue_stats_[tid][symbol_id][id].event_type_ = event_type;
    worker_enqueue_stats_id_[tid][symbol_id]++;
  }

  inline void LogDequeueStatsWorker(int tid, size_t symbol_id, size_t start_tsc,
                                    size_t end_tsc, EventType event_type) {
    size_t id = worker_dequeue_stats_id_[tid][symbol_id];
    worker_dequeue_stats_[tid][symbol_id][id].tsc_start_ = start_tsc;
    worker_dequeue_stats_[tid][symbol_id][id].tsc_end_ = end_tsc;
    worker_dequeue_stats_[tid][symbol_id][id].event_type_ = event_type;
    worker_dequeue_stats_id_[tid][symbol_id]++;
  }

  // Task enqueue/dequeue start and end timestamps and task type
  struct QueueTsStat {
    EventType event_type_;
    size_t tsc_start_ = 0;  // Unit = TSC cycles
    size_t tsc_end_ = 0;    // Unit = TSC cycles
  };

  std::array<std::array<QueueTsStat, kMaxLoggingEventsMaster>, kMaxSymbols>
      enqueue_stats_;
  std::array<QueueTsStat, kMaxLoggingEventsMaster> dequeue_stats_;
  std::array<size_t, kMaxSymbols> enqueue_stats_id_ = {};
  size_t dequeue_stats_id_ = 0;

  std::array<std::array<size_t, kNumStatsFrames>, kMaxThreads>
      total_worker_dequeue_tsc_ = {};
  std::array<std::array<size_t, kNumStatsFrames>, kMaxThreads>
      total_worker_enqueue_tsc_ = {};
  std::array<std::array<size_t, kNumStatsFrames>, kMaxThreads>
      total_worker_valid_dequeue_tsc_ = {};
  std::array<std::array<size_t, kNumStatsFrames>, kMaxThreads>
      worker_num_valid_enqueue_ = {};

  std::array<
      std::array<std::array<QueueTsStat, kMaxLoggingEventsWorker>, kMaxSymbols>,
      kMaxThreads>
      worker_enqueue_stats_;
  std::array<
      std::array<std::array<QueueTsStat, kMaxLoggingEventsWorker>, kMaxSymbols>,
      kMaxThreads>
      worker_dequeue_stats_;
  std::array<std::array<size_t, kMaxSymbols>, kMaxThreads>
      worker_enqueue_stats_id_ = {};
  std::array<std::array<size_t, kMaxSymbols>, kMaxThreads>
      worker_dequeue_stats_id_ = {};

 private:
  void Print() const;
  nlohmann::json Parse(const nlohmann::json& in_json,
                       const std::string& json_handle);

  /* Class constants */
  inline static const size_t kDefaultSymbolNumPerFrame = 70;
  inline static const size_t kDefaultFreqOrthPilotSymbolNum = 1;
  inline static const size_t kDefaultULSymPerFrame = 30;
  inline static const size_t kDefaultULSymStart = 9;
  inline static const size_t kDefaultDLSymPerFrame = 30;
  inline static const size_t kDefaultDLSymStart = 40;

  /* Private class variables */
  const double freq_ghz_;  // RDTSC frequency in GHz

  size_t bs_ant_num_;  // Total number of BS antennas
  size_t bf_ant_num_;  // Number of antennas used in beamforming

  // The count of ues an instance is responsable for
  size_t ue_num_;
  // The count of ue antennas an instance is responsable for
  size_t ue_ant_num_;
  // Feature to adapt the number of ues from 1 to ue_ant_num_
  // across frames_to_test_ frames
  bool adapt_ues_;

  // Total number of us antennas in this experiment including the ones
  // instantiated on other runs/machines.
  size_t ue_ant_total_;
  // The offset into the number total ue antennas this instantiation is
  // responsible for.
  size_t ue_ant_offset_;

  // The total number of OFDM subcarriers, which is a power of two
  size_t ofdm_ca_num_;

  // The number of cyclic prefix IQ samples. These are taken from the tail of
  // the time-domain OFDM samples and prepended to the beginning.
  size_t cp_len_;

  // The number of OFDM subcarriers that are non-zero in the frequency domain
  size_t ofdm_data_num_;

  // The index of the first non-zero OFDM subcarrier (in the frequency domain)
  // in block of ofdm_ca_num_ subcarriers.
  size_t ofdm_data_start_;

  // The index of the last non-zero OFDM subcarrier (in the frequency domain)
  // in block of ofdm_ca_num_ subcarriers.
  size_t ofdm_data_stop_;

  size_t ofdm_pilot_spacing_;

  bool scramble_enabled_;

  nlohmann::json ul_mcs_params_;  // Uplink Modulation and Coding (MCS)
  nlohmann::json dl_mcs_params_;  // Downlink Modulation and Coding (MCS)

  // A class that holds the frame configuration the id contains letters
  // representing the symbol types in the frame (e.g., 'P' for pilot symbols,
  // 'U' for uplink data symbols)
  FrameStats frame_;

  MacUtils mac_params_;

  std::atomic<bool> running_;

  size_t dl_packet_length_;  // HAS_TIME & END_BURST, fixme

  std::vector<SubcarrierType> ul_symbol_map_;
  std::vector<SubcarrierType> dl_symbol_map_;
  std::vector<SubcarrierType> control_symbol_map_;
  std::vector<size_t> dl_symbol_data_id_;
  std::vector<size_t> dl_symbol_ctrl_id_;

  Table<int8_t> ul_mod_bits_;
  Table<int8_t> dl_mod_bits_;
  Table<complex_float> dl_iq_f_;
  Table<complex_float> ul_iq_f_;
  Table<std::complex<int16_t>> dl_iq_t_;
  Table<std::complex<int16_t>> ul_iq_t_;

  std::vector<std::complex<float>> gold_cf32_;
  std::vector<std::complex<int16_t>> beacon_ci16_;
  std::vector<uint32_t> coeffs_;

  /// I/Q samples of common pilot
  std::vector<std::complex<int16_t>> pilot_ci16_;

  std::vector<std::complex<float>> pilot_cf32_;

  /// I/Q samples of pilots per UE antenna per pilot symbol
  std::vector<std::vector<std::vector<std::complex<int16_t>>>> pilot_ue_ci16_;

  /// List of subcarriers used per UE to transmit pilot
  std::vector<arma::uvec> pilot_ue_sc_;

  std::vector<uint32_t> pilot_;
  std::vector<uint32_t> beacon_;
  complex_float* pilots_{nullptr};
  complex_float* pilots_sgn_{nullptr};
  complex_float* pilot_pre_ifft_{nullptr};
  complex_float* pilot_ifft_{nullptr};
  Table<complex_float> ue_specific_pilot_;
  Table<complex_float> ue_pilot_pre_ifft_;
  Table<complex_float> ue_pilot_ifft_;
  Table<std::complex<int16_t>> ue_specific_pilot_t_;
  std::vector<std::complex<float>> common_pilot_;

  std::vector<double> client_gain_tx_a_;
  std::vector<double> client_gain_tx_b_;
  std::vector<double> client_gain_rx_a_;
  std::vector<double> client_gain_rx_b_;

  std::vector<std::string> radio_id_;
  std::vector<std::string> hub_id_;
  std::vector<std::string> ue_radio_id_;
  std::vector<std::string> ue_radio_name_;
  std::vector<size_t> ref_radio_;
  std::vector<size_t> ref_ant_;
  std::vector<size_t> cell_id_;

  // Controls whether the synchronization and frame time keeping is done
  // in hardware or software
  // true: use hardware correlator; false: use software corrleator
  bool hw_framer_;
  bool ue_hw_framer_;
  size_t ue_resync_period_;

  double freq_;
  double rate_;
  double nco_;
  double radio_rf_freq_;
  double bw_filter_;
  bool single_gain_;
  double tx_gain_a_;
  double rx_gain_a_;
  double tx_gain_b_;
  double rx_gain_b_;
  double calib_tx_gain_a_;
  double calib_tx_gain_b_;
  std::vector<double> client_tx_gain_a_;
  std::vector<double> client_rx_gain_a_;
  std::vector<double> client_tx_gain_b_;
  std::vector<double> client_rx_gain_b_;

  size_t num_cells_;
  size_t num_radios_;
  size_t num_channels_;
  size_t num_ue_channels_;
  size_t beacon_ant_;
  size_t beacon_len_;
  size_t init_calib_repeat_;
  bool dynamic_core_allocation_;
  bool smooth_calib_;
  bool beamsweep_;
  bool sample_cal_en_;
  bool imbalance_cal_en_;
  size_t beamforming_algo_;
  size_t num_spatial_streams_;
  size_t vpp_num_replicas_;
  size_t vpp_runs_;
  size_t vpp_iters_;
  size_t vpp_neighbor_window_;
  double vpp_beta_min_;
  double vpp_beta_max_;
  double vpp_lll_delta_;
  int vpp_lll_window_;
  size_t vpp_fcse_full_layers_;
  bool vpp_power_norm_;
  bool vpp_periodic_llr_;
  double vpp_babai_skip_margin_;
  double vpp_fixed_gain_;
  bool tx_peak_agc_;
  double tx_fixed_gain_;
  std::string tx_exact_gain_file_;
  double tx_energy_target_;
  size_t vpp_early_stop_patience_;
  size_t vpp_run_patience_;
  bool vpp_disable_lll_;
  bool vpp_disable_babai_start_;
  std::string beamforming_str_;
  std::vector<bool> external_ref_node_;
  std::string channel_;
  std::string ue_channel_;

  std::vector<size_t> excluded_;
  size_t core_offset_;
  size_t worker_thread_num_;
  size_t socket_thread_num_;
  size_t fft_thread_num_;
  size_t demul_thread_num_;
  size_t decode_thread_num_;
  size_t beam_thread_num_;

  size_t ue_core_offset_;
  size_t ue_worker_thread_num_;
  size_t ue_socket_thread_num_;

  // Number of OFDM data subcarriers handled in one demodulation event
  size_t demul_block_size_;
  // Derived from demul_block_size
  size_t demul_events_per_symbol_;

  /// Number of OFDM data subcarriers handled in 1 kBeam event
  size_t beam_block_size_;
  /// Beam Events generated per Frame.  Derived from beam_block_size
  size_t beam_events_per_symbol_;

  // Number of antennas handled in one FFT event
  size_t fft_block_size_;

  // Number of code blocks handled in one encode event
  size_t encode_block_size_;

  // Whether to enable frequency orthogonal pilot
  bool freq_orthogonal_pilot_;

  // Frequency orthogonal pilot subcarrier group size
  size_t pilot_sc_group_size_;

  // The number of zero IQ samples prepended to a time-domain symbol (i.e.,
  // before the cyclic prefix) before transmission. Its value depends on
  // over-the-air and RF delays, and is currently calculated by manual tuning.
  size_t ofdm_tx_zero_prefix_;

  // The number of zero IQ samples appended to a time-domain symbol before
  // transmission. Its value depends on over-the-air and RF delays, and is
  // currently calculated by manual tuning.
  size_t ofdm_tx_zero_postfix_;

  // The number of IQ samples to skip from the beginning of symbol received by
  // Agora on the uplink at the base station. Due to over-the-air and RF
  // delays, this can be different from (prefix + cp_len_), and is currently
  // calculated by manual tuning.
  size_t ofdm_rx_zero_prefix_bs_;

  size_t ofdm_rx_zero_prefix_cal_ul_;
  size_t ofdm_rx_zero_prefix_cal_dl_;

  // The number of IQ samples to skip from the beginning of symbol received by
  // Agora on the downlink at the client. Due to over-the-air and RF
  // delays, this can be different from (prefix + cp_len_), and is currently
  // calculated by manual tuning.
  size_t ofdm_rx_zero_prefix_client_;

  // The total number of IQ samples in one physical layer time-domain packet
  // received or sent by Agora
  size_t samps_per_symbol_;

  // The number of bytes in one physical layer time-domain packet received or
  // sent by Agora. This includes Agora's packet header, but not the
  // Ethernet/IP/UDP headers.
  size_t packet_length_;

  std::vector<int> cl_tx_advance_;
  std::vector<float> cl_corr_scale_;

  float scale_;  // Scaling factor for all transmit symbols

  bool bigstation_mode_;  // If true, use pipeline-parallel scheduling

  // IP address of the machine running the baseband processing for UE
  std::string ue_server_addr_;

  // IP address of the machine running the baseband processing for BS
  std::string bs_server_addr_;

  // IP address of the base station RRU, RRU emulator (sender),
  // or channel simulator
  std::string bs_rru_addr_;

  // IP address of the Ue RRU, RRU emulator (sender), or channel simulator
  // could be multiple addresses
  std::string ue_rru_addr_;

  // IP address of the data source/sink server communicating with MAC (BS/UE)
  std::string mac_remote_addr_;

  // IP address of the listening server for runtime stats log
  std::string log_listener_addr_;

  int bs_server_port_;  // Base UDP port used by BS to receive data

  // Base RRU/channel simulator UDP port used by BS to transmit downlink data
  int bs_rru_port_;

  int ue_server_port_;  // Base UDP port used by UEs to receive data

  // Base RRU/channel simulator UDP port used by UEs to transmit uplink data
  int ue_rru_port_;

  // Number of NIC ports used for DPDK
  uint16_t dpdk_num_ports_;

  // Offset of the first NIC port used by Agora's DPDK mode
  uint16_t dpdk_port_offset_;

  // MAC addresses of NIC ports separated by ';'
  std::string dpdk_mac_addrs_;

  // Port ID at BaseStation MAC layer side
  size_t bs_mac_rx_port_;
  size_t bs_mac_tx_port_;

  // Port ID at Client MAC layer side
  size_t ue_mac_rx_port_;
  size_t ue_mac_tx_port_;

  // IP Address of the data generation app at the UE, e.g. video streamer
  std::string ue_app_rx_addr_;

  // IP Address of the data generation app at the BS, e.g. video streamer
  std::string bs_app_rx_addr_;

  // Base UDP port used by the data generation app at the UE, e.g. video streamer
  size_t ue_app_rx_port_;

  // Base UDP port used by the data generation app at the BS, e.g. video streamer
  size_t bs_app_rx_port_;

  // Port ID at RP
  std::string rp_remote_host_name_;
  size_t rp_rx_port_;
  size_t rp_tx_port_;

  // Port ID at log listening server
  size_t log_listener_port_;

  // Number of logged subcarrier data samples
  size_t log_sc_num_;

  // Whether use unique timestamp as subdirectory of csv log files
  bool log_timestamp_;

  // Number of frames_ sent by sender during testing = number of frames_
  // processed by Agora before exiting.
  size_t frames_to_test_;

  // Frame number for which the timestamps of different tasks are logged for profiling
  size_t frame_to_profile_;

  // Size of tranport block given by upper layer
  size_t transport_block_size_;

  float noise_level_;
  int noise_mode_;
  size_t distinct_data_frames_;
  bool noiseless_pilots_;
  bool dl_noiseless_pilots_;
  uint64_t channel_seed_;

  const std::string config_filename_;
  std::string trace_file_;
  std::string timestamp_;
  std::vector<std::string> ul_tx_f_data_files_;

  // If true, channel matrix H will be applied in the frequency domain
  bool freq_domain_channel_;

  std::string scheduler_type_;
};
#endif /* CONFIG_HPP_ */
