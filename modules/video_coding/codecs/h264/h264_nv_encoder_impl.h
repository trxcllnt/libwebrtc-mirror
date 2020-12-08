/*
 *  Copyright (c) 2020 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 *
 */

#ifndef MODULES_VIDEO_CODING_CODECS_H264_H264_NV_ENCODER_IMPL_H_
#define MODULES_VIDEO_CODING_CODECS_H264_H264_NV_ENCODER_IMPL_H_

// Everything declared in this header is only required when WebRTC is
// build with NVENC H264 support, please do not move anything out of the
// #ifdef unless needed and tested.
#ifdef WEBRTC_USE_H264_NVENC

#include "api/video/i420_buffer.h"
#include "common_video/h264/h264_bitstream_parser.h"
#include "modules/video_coding/codecs/h264/NvEncoder/NvEncoderCuda.h"
#include "modules/video_coding/codecs/h264/include/h264.h"
#include "third_party/openh264/src/codec/api/svc/codec_app_def.h"

namespace webrtc {

class H264NvEncoderImpl : public H264Encoder {
 public:
  explicit H264NvEncoderImpl(const cricket::VideoCodec& codec);
  ~H264NvEncoderImpl() override;

  int32_t InitEncode(const VideoCodec* codec_settings,
                     const VideoEncoder::Settings& settings) override;
  int32_t Release() override;

  int32_t RegisterEncodeCompleteCallback(
      EncodedImageCallback* callback) override;
  void SetRates(const RateControlParameters& parameters) override;

  // The result of encoding - an EncodedImage and CodecSpecificInfo - are
  // passed to the encode complete callback.
  int32_t Encode(const VideoFrame& frame,
                 const std::vector<VideoFrameType>* frame_types) override;

  EncoderInfo GetEncoderInfo() const override;

  // Exposed for testing.
  H264PacketizationMode PacketizationModeForTesting() const {
    return packetization_mode_;
  }

 private:
  webrtc::H264BitstreamParser h264_bitstream_parser_;

  // Reports statistics with histograms.
  void ReportInit();
  void ReportError();
  void SetStreamState(size_t stream_idx, bool send_stream);

  CUdevice device_;
  CUcontext context_{nullptr};
  std::vector<bool> stream_states_;
  std::vector<NvEncoderCuda*> encoders_;
  std::vector<std::vector<uint8_t>> output_buffer_{};
  std::vector<EncodedImage> encoded_images_;

  std::vector<SSourcePicture> pictures_;
  std::vector<rtc::scoped_refptr<I420Buffer>> downscaled_buffers_;
  std::vector<uint8_t> tl0sync_limit_;

  VideoCodec codec_;
  H264PacketizationMode packetization_mode_;
  size_t max_payload_size_{0};
  EncodedImageCallback* encoded_image_callback_;

  bool has_reported_init_;
  bool has_reported_error_;
  bool key_frame_request_ = false;
};
}  // namespace webrtc

#endif
#endif
