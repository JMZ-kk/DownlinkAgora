/**
 * @file channel.h
 * @brief Declaration file for the channel class
 */
#ifndef CHANNEL_H_
#define CHANNEL_H_

#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "armadillo"
#include "channel.h"
#include "channel_model.h"
#include "config.h"

class Channel {
 public:
  Channel(const Config* const config, std::string& channel_type,
          std::string& dataset_path);
  ~Channel();

  // Dimensions of fmat_src: ( bscfg->sampsPerSymbol, uecfg->UE_ANT_NUM )
  // frame_id: the frame this symbol belongs to. A flat-fading channel is a
  // property of the frame, not of the moment a packet happens to be processed:
  // a symbol that reaches chsim late (e.g. a downlink data symbol from a slow
  // precoder such as VPP/PREQ) must still see the matrix of its own frame even
  // if the next frame's beacon has already drawn a new one.
  // noise_free: apply the channel but skip the AWGN (used for pilot symbols
  // when the "noiseless_pilots" config option is set).
  // is_pilot: whether this symbol is a pilot symbol. Unused by Awgn() --
  // noise is applied uncompensated (no NoiseMode()-based normalization),
  // unlike test_vector_generator.cc's offline synthetic uplink test vectors.
  void ApplyChan(const arma::cx_fmat& fmat_src, arma::cx_fmat& mat_dst,
                 const bool is_downlink, const bool is_newChan,
                 const size_t frame_id, const bool noise_free = false,
                 const bool is_pilot = false);

  // Additive White Gaussian Noise. Dimensions of src: ( bscfg->sampsPerSymbol,
  // uecfg->UE_ANT_NUM )
  void Awgn(const arma::cx_fmat& fmat_src, arma::cx_fmat& fmat_dst,
            const bool noise_free = false, const bool is_downlink = false,
            const bool is_pilot = false) const;

 private:
  // Flat-fading matrices (uplink and downlink orientation) of the most recent
  // frames, indexed by frame id % size(). Guarded by frame_channels_mutex_,
  // which also serializes access to the channel model's own state.
  struct FrameChannel {
    size_t frame_id = std::numeric_limits<size_t>::max();
    arma::cx_fmat h_ul;
    arma::cx_fmat h_dl;
  };

  const Config* const cfg_;
  std::string sim_chan_model_;
  double noise_samp_std_;

  std::vector<FrameChannel> frame_channels_;
  std::mutex frame_channels_mutex_;

  std::unique_ptr<ChannelModel> channel_model_;
};

#endif  // CHANNEL_H_
