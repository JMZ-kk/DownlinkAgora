/**
 * @file channel.cc
 * @brief Implementation file for the channel class
 */
#include "channel.h"

#include "armadillo"
#include "logger.h"
#include "utils.h"

static constexpr bool kPrintChannelOutput = false;
static constexpr bool kPrintSNRCheck = false;

Channel::~Channel() = default;

Channel::Channel(const Config* const config, std::string& in_channel_type,
                 std::string& dataset_path)
    : cfg_(config), sim_chan_model_(std::move(in_channel_type)) {
  channel_model_ = std::move(
      ChannelModel::CreateChannelModel(cfg_, sim_chan_model_, dataset_path));

  noise_samp_std_ = config->NoiseLevel() / std::sqrt(2);
  frame_channels_.resize(kFrameWnd);

  std::cout << "Noise level to be used is: " << std::fixed << std::setw(5)
            << std::setprecision(3) << noise_samp_std_ << std::endl;
}

void Channel::ApplyChan(const arma::cx_fmat& fmat_src, arma::cx_fmat& fmat_dst,
                        const bool is_downlink, const bool is_newChan,
                        const size_t frame_id, const bool noise_free,
                        const bool is_pilot) {
  arma::cx_fmat fmat_h;
  const bool is_flat =
      (channel_model_->GetFadingType() == ChannelModel::kFlat);

  if (is_newChan) {
    if (is_flat) {
      // Draw the frame's matrix once and remember it under the frame id.
      std::lock_guard<std::mutex> lock(frame_channels_mutex_);
      channel_model_->UpdateModel();
      FrameChannel& slot = frame_channels_[frame_id % frame_channels_.size()];
      slot.frame_id = frame_id;
      slot.h_ul = channel_model_->GetMatrix(false);
      slot.h_dl = channel_model_->GetMatrix(true);
    } else {
      channel_model_->UpdateModel();
    }
  }

  switch (channel_model_->GetFadingType()) {
    case ChannelModel::kFlat: {
      arma::cx_fmat h;
      {
        std::lock_guard<std::mutex> lock(frame_channels_mutex_);
        const FrameChannel& slot =
            frame_channels_[frame_id % frame_channels_.size()];
        if (slot.frame_id == frame_id) {
          h = is_downlink ? slot.h_dl : slot.h_ul;
        } else {
          // No beacon has been seen for this frame (or it was evicted): fall
          // back to the model's current matrix.
          h = channel_model_->GetMatrix(is_downlink);
        }
      }
      fmat_h = fmat_src * h;
      break;
    }

    case ChannelModel::kSelective: {
      const size_t n_rows = (cfg_->FreqDomainChannel())
                                ? cfg_->OfdmCaNum()
                                : cfg_->SampsPerSymbol();
      const size_t n_cols = (is_downlink) ? cfg_->UeAntNum() : cfg_->BsAntNum();

      fmat_h.zeros(n_rows, n_cols);
      for (size_t h_index = 0; h_index < n_rows; h_index++) {
        fmat_h.row(h_index) = fmat_src.row(h_index) *
                              channel_model_->GetMatrix(is_downlink, h_index);
      }
      break;
    }

    default: {
      AGORA_LOG_ERROR("Invalid Channel model fading type \n");
      break;
    }
  }

  // Add noise
  Awgn(fmat_h, fmat_dst, noise_free, is_downlink, is_pilot);

  if (kPrintChannelOutput) {
    Utils::PrintMat(fmat_dst, "H");
  }
}

void Channel::Awgn(const arma::cx_fmat& src, arma::cx_fmat& dst,
                   const bool noise_free, const bool is_downlink,
                   [[maybe_unused]] const bool is_pilot) const {
  if (noise_free || cfg_->NoiseLevel() < 0.0001f) {
    // Noise level negligible (or noise explicitly disabled for this symbol):
    // skip the RNG work entirely.
    dst = src;
  } else {
    const int n_row = src.n_rows;
    const int n_col = src.n_cols;

    // Generate noise
    arma::cx_fmat noise(arma::randn<arma::fmat>(n_row, n_col),
                        arma::randn<arma::fmat>(n_row, n_col));

    noise *= noise_samp_std_;

    // Uncompensated: noise_samp_std_ (derived from noise_level alone) is
    // applied as-is regardless of direction, UE count, or NoiseMode() --
    // unlike test_vector_generator.cc's offline uplink test vectors, this
    // live channel path does not normalize for active-UE-count, so SNR is
    // free to grow with UE count (uplink) or shift with array/precoder gain
    // (downlink) exactly as a real receiver with a fixed noise floor would
    // see.
    dst = src + noise;

    // Check SNR
    if (kPrintSNRCheck) {
      arma::fmat noise_sq = arma::square(abs(noise));
      arma::frowvec noise_vec = arma::mean(noise_sq, 0);
      arma::fmat src_sq = arma::square(abs(src));
      arma::frowvec pwr_vec = arma::mean(src_sq, 0);
      arma::frowvec snr = 10.0f * arma::log10(pwr_vec / noise_vec);
      std::cout << (is_downlink ? "DL SNR: " : "UL SNR: ") << snr;
    }
  }
}
