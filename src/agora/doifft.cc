/**
 * @file doifft.cc
 * @brief Implementation file for the DoIFFT class.
 */
#include "doifft.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include "comms-lib.h"
#include "concurrent_queue_wrapper.h"
#include "datatype_conversion.h"
#include "logger.h"

static constexpr bool kPrintIFFTOutput = false;
static constexpr bool kPrintSocketOutput = false;
static constexpr bool kUseOutOfPlaceIFFT = false;
static constexpr bool kMemcpyBeforeIFFT = true;
static constexpr bool kPrintIfftStats = false;

namespace {
// Burst-level peak-amplitude AGC (Config::TxPeakAgc(), see config.h): ONE
// shared gain for every antenna and symbol in a frame, re-derived from the
// previous frame's own fully-observed peak -- Agora streams frames in real
// time and can't see a frame's true peak before transmitting it (unlike
// paramax_ORBIT's offline ofdm_modulate.h, which scans the whole burst
// before applying one gain to it), so this is a 1-frame-lagged
// approximation of the same idea rather than an exact replica. The gain
// must be identical across every antenna of a symbol (and every symbol of
// the frame): per-antenna or per-symbol scaling would break the precoder's
// H*P~=I invariant (paramax_ORBIT dl_precode.h's own warning) -- an
// earlier version of this mutated a shared running estimate on every
// single (frame,symbol,antenna) call, which gave concurrently-running
// antennas of the same symbol different gains and corrupted beamforming
// coherence outright (measured: ~65-90% BLER for BOTH Zf and Vpp, far more
// damage than the handful of clipping events alone could cause). Freezing
// the gain once per frame transition, used by every call until the next
// frame's first call arrives, avoids that.
constexpr float kTxAgcTargetPeak = 0.6f;  // full scale is 1.0; conservative since this lags a whole frame, not a true pre-scan
std::mutex g_tx_agc_mutex;
size_t g_tx_agc_frame = SIZE_MAX;         // frame this gain was frozen for
float g_tx_agc_gain = 1.0f;               // frozen gain, shared by the whole frame
float g_tx_agc_prev_frame_peak = kTxAgcTargetPeak;  // basis for the next freeze
float g_tx_agc_accum_peak = 0.0f;         // this (in-progress) frame's own peak so far

// Returns the one gain every antenna/symbol of frame_id must use. The
// first call seen for a new frame_id freezes it (from the previous frame's
// fully-accumulated peak) and resets the accumulator; every later call for
// the same frame_id reuses that frozen value untouched.
float TxAgcGetGain(size_t frame_id) {
  std::lock_guard<std::mutex> lk(g_tx_agc_mutex);
  if (frame_id != g_tx_agc_frame) {
    if (g_tx_agc_frame != SIZE_MAX) {
      g_tx_agc_prev_frame_peak = std::max(g_tx_agc_accum_peak, 1e-6f);
    }
    g_tx_agc_accum_peak = 0.0f;
    g_tx_agc_frame = frame_id;
    g_tx_agc_gain = kTxAgcTargetPeak / g_tx_agc_prev_frame_peak;
  }
  return g_tx_agc_gain;
}

// Folds this call's own raw (gain-backed-out) peak into the current
// frame's accumulator, for TxAgcGetGain() to freeze once this frame ends.
void TxAgcObservePeak(float raw_peak) {
  std::lock_guard<std::mutex> lk(g_tx_agc_mutex);
  g_tx_agc_accum_peak = std::max(g_tx_agc_accum_peak, raw_peak);
}

// Config::TxExactGainFile(): the TRUE, non-lagged optimal per-symbol gain
// c* = target/true_peak (see the "can c be written as a formula" answer --
// true_peak has no closed form, it's measured), loaded from a file built
// from an earlier gain=1 run of the SAME data_generator output (same
// payload bits). "frame,dl_symbol_idx,gain" rows, no header.
std::mutex g_tx_exact_gain_mutex;
bool g_tx_exact_gain_loaded = false;
std::unordered_map<size_t, float> g_tx_exact_gain_table;  // key: frame*4096+dl_symbol_idx

float TxExactGain(const std::string& path, size_t frame_id,
                  size_t dl_symbol_idx) {
  std::lock_guard<std::mutex> lk(g_tx_exact_gain_mutex);
  if (!g_tx_exact_gain_loaded) {
    g_tx_exact_gain_loaded = true;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
      std::istringstream ss(line);
      std::string frame_s, sym_s, gain_s;
      if (std::getline(ss, frame_s, ',') && std::getline(ss, sym_s, ',') &&
          std::getline(ss, gain_s)) {
        const size_t frm = std::stoull(frame_s);
        const size_t sym = std::stoull(sym_s);
        const float gain = std::stof(gain_s);
        g_tx_exact_gain_table[frm * 4096 + sym] = gain;
      }
    }
    AGORA_LOG_ERROR("TxExactGain: loaded %zu entries from %s\n",
                    g_tx_exact_gain_table.size(), path.c_str());
  }
  const auto it = g_tx_exact_gain_table.find(frame_id * 4096 + dl_symbol_idx);
  return (it != g_tx_exact_gain_table.end()) ? it->second : 1.0f;
}
}  // namespace

DoIFFT::DoIFFT(Config* in_config, int in_tid,
               Table<complex_float>& in_dl_ifft_buffer,
               char* in_dl_socket_buffer, Stats* in_stats_manager,
               VppPowerStats& vpp_power)
    : Doer(in_config, in_tid),
      dl_ifft_buffer_(in_dl_ifft_buffer),
      dl_socket_buffer_(in_dl_socket_buffer),
      vpp_power_(vpp_power) {
  duration_stat_ = in_stats_manager->GetDurationStat(DoerType::kIFFT, in_tid);
  DftiCreateDescriptor(&mkl_handle_, DFTI_SINGLE, DFTI_COMPLEX, 1,
                       cfg_->OfdmCaNum());
  if (kUseOutOfPlaceIFFT) {
    DftiSetValue(mkl_handle_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  }
  DftiCommitDescriptor(mkl_handle_);

  // Aligned for SIMD
  ifft_out_ = static_cast<float*>(
      Agora_memory::PaddedAlignedAlloc(Agora_memory::Alignment_t::kAlign64,
                                       2 * cfg_->OfdmCaNum() * sizeof(float)));

  ifft_shift_tmp_ = static_cast<complex_float*>(
      Agora_memory::PaddedAlignedAlloc(Agora_memory::Alignment_t::kAlign64,
                                       2 * cfg_->OfdmCaNum() * sizeof(float)));
  ifft_scale_factor_ = cfg_->OfdmCaNum();
}

DoIFFT::~DoIFFT() {
  DftiFreeDescriptor(&mkl_handle_);
  std::free(ifft_out_);
  std::free(ifft_shift_tmp_);
}

EventData DoIFFT::Launch(size_t tag) {
  size_t start_tsc = GetTime::WorkerRdtsc();

  const size_t frame_id = gen_tag_t(tag).frame_id_;
  const size_t symbol_id = gen_tag_t(tag).symbol_id_;
  const size_t ant_id = gen_tag_t(tag).ant_id_;
  const bool bypass_ifft = cfg_->FreqDomainChannel();

  if (kDebugPrintInTask) {
    std::printf("In doIFFT thread %d: frame: %zu, symbol: %zu, antenna: %zu\n",
                tid_, frame_id, symbol_id, ant_id);
  }

  const size_t dl_symbol_idx = cfg_->Frame().GetDLSymbolIdx(symbol_id);
  const size_t total_symbol_idx_dl =
      cfg_->GetTotalSymbolIdxDl(frame_id, dl_symbol_idx);
  const size_t in_offset = (total_symbol_idx_dl * cfg_->BsAntNum()) + ant_id;

  const size_t total_symbol_idx =
      cfg_->GetTotalSymbolIdxDlBcast(frame_id, symbol_id);
  const size_t out_offset = (total_symbol_idx * cfg_->BsAntNum()) + ant_id;

  const size_t start_tsc1 = GetTime::WorkerRdtsc();
  duration_stat_->task_duration_[1u] += start_tsc1 - start_tsc;

  auto* ifft_in_ptr = reinterpret_cast<float*>(dl_ifft_buffer_[in_offset]);
  auto* ifft_out_ptr =
      (kUseOutOfPlaceIFFT || kMemcpyBeforeIFFT) ? ifft_out_ : ifft_in_ptr;

  // Highest priority: the true, measured, non-lagged per-symbol optimum
  // (see Config::TxExactGainFile()).
  const bool exact_gain_active = !cfg_->TxExactGainFile().empty() &&
                                  cfg_->Frame().GetSymbolType(symbol_id) ==
                                      SymbolType::kDL;
  // Next: rescale THIS algorithm's own actual precoded energy to a fixed,
  // P/u-independent external constant (see Config::TxEnergyTarget()) --
  // applies to any algorithm, unlike VppPowerNorm below which only matches
  // Vpp to Zf's own (also P/u-dependent) energy.
  const bool energy_target_active =
      !exact_gain_active && cfg_->TxEnergyTarget() > 0.0 &&
      cfg_->Frame().GetSymbolType(symbol_id) == SymbolType::kDL &&
      dl_symbol_idx >= cfg_->Frame().ClientDlPilotSymbols();
  if (exact_gain_active) {
    const auto gain = TxExactGain(cfg_->TxExactGainFile(), frame_id,
                                  dl_symbol_idx);
    for (size_t i = cfg_->OfdmDataStart() * 2; i < cfg_->OfdmDataStop() * 2;
         i++) {
      ifft_in_ptr[i] *= gain;
    }
  } else if (energy_target_active) {
    const auto gain = static_cast<float>(vpp_power_.GainToTarget(
        frame_id, dl_symbol_idx, cfg_->TxEnergyTarget()));
    for (size_t i = cfg_->OfdmDataStart() * 2; i < cfg_->OfdmDataStop() * 2;
         i++) {
      ifft_in_ptr[i] *= gain;
    }
  } else if (cfg_->TxFixedGain() > 0.0 &&
             cfg_->Frame().GetSymbolType(symbol_id) == SymbolType::kDL) {
    // A plain fixed multiplier, independent of VppPowerNorm/Vpp entirely --
    // see Config::TxFixedGain(). Can be tested in isolation on any
    // algorithm, including plain ZF.
    const auto gain = static_cast<float>(cfg_->TxFixedGain());
    for (size_t i = cfg_->OfdmDataStart() * 2; i < cfg_->OfdmDataStop() * 2;
         i++) {
      ifft_in_ptr[i] *= gain;
    }
  }

  // Vpp power normalization: a data symbol carries the same total energy
  // plain ZF would have sent (see vpp_power_stats.h). The UE recovers this
  // gain from the DMRS tones of the same symbol. Pilot symbols stay unscaled.
  float tx_agc_gain = 1.0f;
  const bool tx_agc_active = !exact_gain_active && !energy_target_active &&
                              cfg_->TxFixedGain() <= 0.0 &&
                              cfg_->TxPeakAgc() &&
                              cfg_->Frame().GetSymbolType(symbol_id) == SymbolType::kDL;
  if (tx_agc_active) {
    // Applies to every beamforming algorithm (not Vpp-specific) and to
    // pilot and data symbols alike, mirroring paramax_ORBIT's shared
    // burst-wide gain covering its TDM-LTF training slots and payload
    // symbols together. See TxAgcGetGain()/TxAgcObservePeak() above.
    tx_agc_gain = TxAgcGetGain(frame_id);
    for (size_t i = cfg_->OfdmDataStart() * 2; i < cfg_->OfdmDataStop() * 2;
         i++) {
      ifft_in_ptr[i] *= tx_agc_gain;
    }
  } else if (!exact_gain_active && !energy_target_active &&
             cfg_->TxFixedGain() <= 0.0 &&
             cfg_->VppPowerNorm() &&
             CommsLib::IsPerturbationPrecoder(cfg_->BeamformingAlgo()) &&
             cfg_->Frame().GetSymbolType(symbol_id) == SymbolType::kDL &&
             dl_symbol_idx >= cfg_->Frame().ClientDlPilotSymbols()) {
    const double fixed_gain = cfg_->VppFixedGain();
    const auto gain = static_cast<float>(
        (fixed_gain > 0.0) ? fixed_gain
                           : vpp_power_.Gain(frame_id, dl_symbol_idx));
    for (size_t i = cfg_->OfdmDataStart() * 2; i < cfg_->OfdmDataStop() * 2;
         i++) {
      ifft_in_ptr[i] *= gain;
    }
  }

  std::memset(ifft_in_ptr, 0, sizeof(float) * cfg_->OfdmDataStart() * 2);
  std::memset(ifft_in_ptr + (cfg_->OfdmDataStop()) * 2, 0,
              sizeof(float) * cfg_->OfdmDataStart() * 2);

  if (bypass_ifft && kMemcpyBeforeIFFT) {
    std::memcpy(ifft_out_ptr, ifft_in_ptr,
                sizeof(float) * cfg_->OfdmCaNum() * 2);
  } else {
    CommsLib::FFTShift(reinterpret_cast<complex_float*>(ifft_in_ptr),
                       ifft_shift_tmp_, cfg_->OfdmCaNum());
    if (kMemcpyBeforeIFFT) {
      std::memcpy(ifft_out_ptr, ifft_in_ptr,
                  sizeof(float) * cfg_->OfdmCaNum() * 2);
      DftiComputeBackward(mkl_handle_, ifft_out_ptr);
    } else {
      if (kUseOutOfPlaceIFFT) {
        // Use out-of-place IFFT here is faster than in place IFFT
        // There is no need to reset non-data subcarriers in ifft input
        // to 0 since their values are not changed after IFFT
        DftiComputeBackward(mkl_handle_, ifft_in_ptr, ifft_out_ptr);
      } else {
        DftiComputeBackward(mkl_handle_, ifft_in_ptr);
      }
    }
  }

  bool clipping = false;
  float max_abs = 0;
  for (size_t i = 0; i < 2 * cfg_->OfdmCaNum(); i++) {
    float sample_val = ifft_out_ptr[i] / ifft_scale_factor_;
    if (sample_val >= 1) {
      clipping = true;
      break;
    }
    if (std::abs(sample_val) > max_abs) {
      max_abs = std::abs(sample_val);
    }
  }
  if (clipping) {
    AGORA_LOG_WARN(
        "Clipping occured in Frame %zu, Symbol %zu, Antenna "
        "%zu\n",
        frame_id, symbol_id, ant_id);
  }
  // PEAKLOG: temporary measurement pass for the exact-gain-file workflow
  // (see Config::TxExactGainFile()) -- enabled only via env var so it
  // never fires in a normal run. Logs this call's own raw peak at
  // whatever gain was actually applied above (1.0x when none of
  // TxFixedGain/TxPeakAgc/VppPowerNorm/TxExactGainFile are active).
  if (std::getenv("PEAK_MEASURE") != nullptr &&
      cfg_->Frame().GetSymbolType(symbol_id) == SymbolType::kDL) {
    // energy_cap = sqrt(sum e_ref / sum e_vpp) for this symbol -- the
    // largest c that keeps ||c*P(u+tau*l)||^2 <= ||P*u||^2 (see
    // vpp_power_stats.h). Logged alongside the raw peak so a post-process
    // step can take c = min(target/peak, energy_cap), matching exactly:
    // "if ||c*P(u+tau*l)||^2 would exceed ||P*u||^2, cap c at
    // sqrt(||P*u||^2/||P(u+tau*l)||^2) instead."
    const double energy_cap =
        CommsLib::IsPerturbationPrecoder(cfg_->BeamformingAlgo())
            ? vpp_power_.Gain(frame_id, dl_symbol_idx)
            : 1.0;
    AGORA_LOG_ERROR(
        "PEAKLOG frame=%zu symbol=%zu ant=%zu max_abs=%.6f energy_cap=%.6f\n",
        frame_id, dl_symbol_idx, ant_id, clipping ? 1.0f : max_abs,
        energy_cap);
  }
  if (tx_agc_active) {
    // Back out this call's own (frame-shared) gain to recover the raw
    // (pre-gain) peak this symbol would have produced, and fold it into
    // this frame's accumulator for the NEXT frame's gain. The clipping
    // loop above breaks out early without finishing the scan, so max_abs
    // is only a lower bound when clipping was detected -- 1.0f (full
    // scale) is itself then a valid lower bound on the raw peak, same as
    // any other max_abs value here.
    const float raw_peak = (clipping ? 1.0f : max_abs) / tx_agc_gain;
    TxAgcObservePeak(raw_peak);
  }
  if (ant_id < cfg_->BfAntNum() && max_abs < 1e-4) {
    AGORA_LOG_WARN(
        "Possibly bad antenna %zu with max sample value "
        "%2.2f\n",
        ant_id, max_abs);
  }
  if (kPrintIfftStats) {
    std::printf("%2.3f\n", max_abs);
  }

  if (kPrintIFFTOutput) {
    std::stringstream ss;
    ss << "IFFT_output" << ant_id << "=[";
    for (size_t i = 0; i < cfg_->OfdmCaNum(); i++) {
      ss << std::fixed << std::setw(5) << std::setprecision(3)
         << dl_ifft_buffer_[in_offset][i].re << "+1j*"
         << dl_ifft_buffer_[in_offset][i].im << " ";
    }
    ss << "];" << std::endl;
    std::cout << ss.str();
  }

  const size_t start_tsc2 = GetTime::WorkerRdtsc();
  duration_stat_->task_duration_[2u] += start_tsc2 - start_tsc1;

  auto* pkt = reinterpret_cast<Packet*>(
      &dl_socket_buffer_[out_offset * cfg_->DlPacketLength()]);
  short* socket_ptr = &pkt->data_[2u * cfg_->OfdmTxZeroPrefix()];

  // IFFT scaled results by OfdmCaNum(), we scale down IFFT results
  // during data type coversion.  * 2 complex float -> float
  SimdConvertFloatToShort(ifft_out_ptr, socket_ptr, cfg_->OfdmCaNum() * 2,
                          cfg_->CpLen() * 2, ifft_scale_factor_);

  duration_stat_->task_duration_[3u] += GetTime::WorkerRdtsc() - start_tsc2;

  if (kPrintSocketOutput) {
    std::stringstream ss;
    ss << "socket_tx_data" << ant_id << "_" << symbol_id << "=[";
    for (size_t i = 0; i < cfg_->SampsPerSymbol(); i++) {
      ss << socket_ptr[i * 2] << "+1j*" << socket_ptr[i * 2 + 1] << " ";
    }
    ss << "];" << std::endl;
    std::cout << ss.str();
  }

  duration_stat_->task_count_++;
  duration_stat_->task_duration_[0u] += GetTime::WorkerRdtsc() - start_tsc;
  return {EventType::kIFFT, tag};
}
