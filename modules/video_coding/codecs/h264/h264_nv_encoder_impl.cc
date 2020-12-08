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

// Everything declared in this TU is only required when WebRTC is
// build with NVENC H264 support, please do not move anything out of the
// #ifdef unless needed and tested.
#ifdef WEBRTC_USE_H264_NVENC

#include "absl/strings/match.h"
#include "common_video/libyuv/include/webrtc_libyuv.h"
#include "modules/video_coding/codecs/interface/common_constants.h"
#include "modules/video_coding/include/video_error_codes.h"
#include "modules/video_coding/utility/simulcast_rate_allocator.h"
#include "modules/video_coding/utility/simulcast_utility.h"
#include "rtc_base/logging.h"
#include "system_wrappers/include/metrics.h"
#include "third_party/libyuv/include/libyuv/scale.h"

#include "modules/video_coding/codecs/h264/h264_nv_encoder_impl.h"
#include "modules/video_coding/codecs/h264/include/nvEncodeAPI.h"

#include <algorithm>
#include <numeric>

#define H264NVENCODER_CUDA_CALL(expr)                               \
  do {                                                              \
    auto const status = (expr);                                     \
    if (status != CUDA_SUCCESS) {                                   \
      const char* name = NULL;                                      \
      const char* error = NULL;                                     \
      cuGetErrorName(status, &name);                                \
      cuGetErrorString(status, &error);                             \
      RTC_LOG(LS_ERROR) << __FUNCTION__ << ":\n"                    \
                        << "\"" << #expr << "\"\n"                  \
                        << " returned CUDA error " << name << ":\n" \
                        << error << " at\n"                         \
                        << __FILE__ << ":" << __LINE__;             \
      ReportError();                                                \
      return WEBRTC_VIDEO_CODEC_ENCODER_FAILURE;                    \
    }                                                               \
  } while (0)

#define H264NVENCODER_NVENC_CALL(expr)                             \
  do {                                                             \
    auto const status = (expr);                                    \
    if (status != NV_ENC_SUCCESS) {                                \
      RTC_LOG(LS_ERROR) << __FUNCTION__ << ":\n"                   \
                        << "\"" << #expr << "\"\n"                 \
                        << " returned error " << status << " at\n" \
                        << __FILE__ << ":" << __LINE__;            \
      ReportError();                                               \
      return WEBRTC_VIDEO_CODEC_ENCODER_FAILURE;                   \
    }                                                              \
  } while (0)

#define H264NVENCODER_NVENC_CALL_SAFE(expr)                        \
  do {                                                             \
    auto const status = (expr);                                    \
    if (status != NV_ENC_SUCCESS) {                                \
      RTC_LOG(LS_ERROR) << __FUNCTION__ << ":\n"                   \
                        << "\"" << #expr << "\"\n"                 \
                        << " returned error " << status << " at\n" \
                        << __FILE__ << ":" << __LINE__;            \
      ReportError();                                               \
      return;                                                      \
    }                                                              \
  } while (0)

namespace webrtc {

namespace {

// Used by histograms. Values of entries should not be changed.
enum H264EncoderImplEvent {
  kH264EncoderEventInit = 0,
  kH264EncoderEventError = 1,
  kH264EncoderEventMax = 16,
};

// QP scaling thresholds.
static const int kLowH264QpThreshold = 24;
static const int kHighH264QpThreshold = 37;

static bool cudaDriverInitialized = false;

}  // namespace

H264NvEncoderImpl::H264NvEncoderImpl(const cricket::VideoCodec& codec)
    : packetization_mode_(H264PacketizationMode::SingleNalUnit),
      max_payload_size_(0),
      encoded_image_callback_(nullptr),
      has_reported_init_(false),
      has_reported_error_(false) {
  RTC_CHECK(absl::EqualsIgnoreCase(codec.name, cricket::kH264CodecName));
  std::string packetization_mode_string;
  if (codec.GetParam(cricket::kH264FmtpPacketizationMode,
                     &packetization_mode_string) &&
      packetization_mode_string == "1") {
    packetization_mode_ = H264PacketizationMode::NonInterleaved;
  }

  encoders_.reserve(kMaxSimulcastStreams);
  pictures_.reserve(kMaxSimulcastStreams);
  stream_states_.reserve(kMaxSimulcastStreams);
  encoded_images_.reserve(kMaxSimulcastStreams);
  downscaled_buffers_.reserve(kMaxSimulcastStreams - 1);
  // configurations_.reserve(kMaxSimulcastStreams);
  tl0sync_limit_.reserve(kMaxSimulcastStreams);
}

H264NvEncoderImpl::~H264NvEncoderImpl() {
  Release();
}

int32_t H264NvEncoderImpl::InitEncode(const VideoCodec* inst,
                                      const VideoEncoder::Settings& settings) {
  ReportInit();

  if (!cudaDriverInitialized) {
    H264NVENCODER_CUDA_CALL(cuInit(0));
    cudaDriverInitialized = true;
  }

  if (!inst ||                               //
      inst->codecType != kVideoCodecH264 ||  //
      inst->maxFramerate == 0 ||             //
      inst->width < 1 || inst->height < 1) {
    ReportError();
    return WEBRTC_VIDEO_CODEC_ERR_PARAMETER;
  }

  int32_t release_ret = Release();
  if (release_ret != WEBRTC_VIDEO_CODEC_OK) {
    ReportError();
    return release_ret;
  }

  int32_t number_of_streams = SimulcastUtility::NumberOfSimulcastStreams(*inst);
  if (number_of_streams > 1 &&
      !SimulcastUtility::ValidSimulcastParameters(*inst, number_of_streams)) {
    return WEBRTC_VIDEO_CODEC_ERR_SIMULCAST_PARAMETERS_NOT_SUPPORTED;
  }

  encoders_.resize(number_of_streams);
  pictures_.resize(number_of_streams);
  stream_states_.resize(number_of_streams);
  encoded_images_.resize(number_of_streams);
  downscaled_buffers_.resize(number_of_streams - 1);
  // configurations_.resize(number_of_streams);
  tl0sync_limit_.resize(number_of_streams);

  codec_ = *inst;
  max_payload_size_ = settings.max_payload_size;

  // Code expects simulcastStream resolutions to be correct, make sure they are
  // filled even when there are no simulcast layers.
  if (codec_.numberOfSimulcastStreams == 0) {
    codec_.simulcastStream[0].width = codec_.width;
    codec_.simulcastStream[0].height = codec_.height;
  }

  // // Temporal layers not supported.
  // if (codec_.simulcastStream[0].numberOfTemporalLayers > 1) {
  //   Release();
  //   return WEBRTC_VIDEO_CODEC_ERR_SIMULCAST_PARAMETERS_NOT_SUPPORTED;
  // }

  key_frame_request_ = false;

  H264NVENCODER_CUDA_CALL(cuDeviceGet(&device_, 0));
  H264NVENCODER_CUDA_CALL(cuDevicePrimaryCtxRetain(&context_, device_));
  H264NVENCODER_CUDA_CALL(cuCtxPushCurrent(context_));

  for (int i = 0, stream_idx = number_of_streams - 1; i < number_of_streams;
       ++i, --stream_idx) {
    encoders_[i] = new NvEncoderCuda();
    H264NVENCODER_NVENC_CALL(
        encoders_[i]->Initialize(context_,                                   //
                                 codec_.simulcastStream[stream_idx].width,   //
                                 codec_.simulcastStream[stream_idx].height,  //
                                 NV_ENC_BUFFER_FORMAT_IYUV, 0));

    NV_ENC_CONFIG encode_config{NV_ENC_CONFIG_VER};
    NV_ENC_INITIALIZE_PARAMS init_params{NV_ENC_INITIALIZE_PARAMS_VER};
    init_params.encodeConfig = &encode_config;
    H264NVENCODER_NVENC_CALL(encoders_[i]->CreateDefaultEncoderParams(
        &init_params,
        // TODO
        NV_ENC_CODEC_H264_GUID,
        // TODO
        NV_ENC_PRESET_P7_GUID,
        // TODO
        NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY));

    init_params.enableWeightedPrediction = 1;
    encode_config.rcParams.enableLookahead = false;
    encode_config.rcParams.lowDelayKeyFrameScale = 1;
    encode_config.encodeCodecConfig.h264Config.numTemporalLayers =
        codec_.simulcastStream[stream_idx].numberOfTemporalLayers;
    // TODO -- test temporal AQ
    // encode_config.rcParams.enableTemporalAQ = 1;
    encode_config.encodeCodecConfig.h264Config.repeatSPSPPS = 1;
    encode_config.encodeCodecConfig.h264Config.disableSPSPPS = 0;
    encode_config.encodeCodecConfig.h264Config.chromaFormatIDC =
        1;  // 1 for YUV420
    // TODO
    encode_config.profileGUID = NV_ENC_H264_PROFILE_BASELINE_GUID;

    H264NVENCODER_NVENC_CALL(encoders_[i]->CreateEncoder(&init_params));

    // Create downscaled image buffers.
    if (i > 0) {
      downscaled_buffers_[i - 1] = I420Buffer::Create(
          encoders_[i]->GetEncodeWidth(), encoders_[i]->GetEncodeHeight(),
          encoders_[i]->GetEncodeWidth(), encoders_[i]->GetEncodeWidth() / 2,
          encoders_[i]->GetEncodeWidth() / 2);
    }

    // Create the encoded output buffer.
    // Default buffer size: size of unencoded data.

    encoded_images_[i].SetEncodedData(EncodedImageBuffer::Create(
        CalcBufferSize(VideoType::kI420, encoders_[i]->GetEncodeWidth(),
                       encoders_[i]->GetEncodeHeight())));
    encoded_images_[i]._encodedWidth = encoders_[i]->GetEncodeWidth();
    encoded_images_[i]._encodedHeight = encoders_[i]->GetEncodeHeight();
    encoded_images_[i].set_size(encoded_images_[i].GetEncodedData()->size());

    tl0sync_limit_[i] =
        codec_.simulcastStream[stream_idx].numberOfTemporalLayers;
  }

  H264NVENCODER_CUDA_CALL(cuCtxPopCurrent(&context_));

  SimulcastRateAllocator init_allocator(codec_);
  VideoBitrateAllocation allocation =
      init_allocator.Allocate(VideoBitrateAllocationParameters(
          DataRate::KilobitsPerSec(codec_.startBitrate), codec_.maxFramerate));
  SetRates(RateControlParameters(allocation, codec_.maxFramerate));

  return WEBRTC_VIDEO_CODEC_OK;
}

int32_t H264NvEncoderImpl::Release() {
  while (!encoders_.empty()) {
    auto encoder_ = encoders_.back();
    if (encoder_) {
      delete encoder_;
    }
    encoders_.pop_back();
  }
  if (context_) {
    if (device_) {
      cuDevicePrimaryCtxRelease(device_);
    }
    context_ = nullptr;
  }
  key_frame_request_ = false;
  pictures_.clear();
  stream_states_.clear();
  encoded_images_.clear();
  downscaled_buffers_.clear();
  return WEBRTC_VIDEO_CODEC_OK;
}

int32_t H264NvEncoderImpl::RegisterEncodeCompleteCallback(
    EncodedImageCallback* callback) {
  encoded_image_callback_ = callback;
  return WEBRTC_VIDEO_CODEC_OK;
}

void H264NvEncoderImpl::SetRates(const RateControlParameters& parameters) {
  if (encoders_.empty()) {
    RTC_LOG(LS_WARNING) << "SetRates() while uninitialized.";
    return;
  }

  if (parameters.framerate_fps < 1.0) {
    RTC_LOG(LS_WARNING) << "Invalid frame rate: " << parameters.framerate_fps;
    return;
  }

  if (parameters.bitrate.get_sum_bps() == 0) {
    // Encoder paused, turn off all encoding.
    for (size_t i = 0; ++i < stream_states_.size(); ++i) {
      SetStreamState(i, false);
    }
    return;
  }

  codec_.maxFramerate = static_cast<uint32_t>(parameters.framerate_fps);

  for (size_t i = 0, stream_idx = encoders_.size() - 1; i < encoders_.size();
       ++i, --stream_idx) {
    NV_ENC_CONFIG encode_config{NV_ENC_CONFIG_VER};
    NV_ENC_RECONFIGURE_PARAMS reconfig_params{NV_ENC_RECONFIGURE_PARAMS_VER};
    reconfig_params.reInitEncodeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
    reconfig_params.reInitEncodeParams.encodeConfig = &encode_config;
    H264NVENCODER_NVENC_CALL_SAFE(
        encoders_[i]->GetInitializeParams(&reconfig_params.reInitEncodeParams));

    reconfig_params.reInitEncodeParams.frameRateDen = 1;
    reconfig_params.reInitEncodeParams.frameRateNum = codec_.maxFramerate;

    // Enable VBR with Constant Rate Factor (CRF) quality for decent on-the-fly
    // streaming
    encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
    encode_config.rcParams.targetQuality = kLowH264QpThreshold - 1;

    encode_config.rcParams.vbvBufferSize =
        static_cast<uint32_t>(1.05f *  //
                              max_payload_size_ * 8
                              // TODO: Is this a better VBV size?
                              // bitrate * 1 / codec_.maxFramerate
        );
    encode_config.rcParams.vbvInitialDelay =
        encode_config.rcParams.vbvBufferSize;

    encode_config.rcParams.maxBitRate = parameters.bitrate.get_sum_bps();
    encode_config.rcParams.averageBitRate =
        parameters.bitrate.GetSpatialLayerSum(stream_idx);

    SetStreamState(i, encode_config.rcParams.averageBitRate > 0);

    if (stream_states_[i]) {
      switch (packetization_mode_) {
        case H264PacketizationMode::SingleNalUnit:
          encode_config.encodeCodecConfig.h264Config.sliceMode = 1;
          encode_config.encodeCodecConfig.h264Config.sliceModeData =
              max_payload_size_;
          RTC_LOG(INFO) << "Encoder is configured with NALU constraint: "
                        << max_payload_size_ << " bytes";
          encode_config.encodeCodecConfig.h264Config.enableConstrainedEncoding =
              1;
          break;
        case H264PacketizationMode::NonInterleaved:
          encode_config.encodeCodecConfig.h264Config.enableConstrainedEncoding =
              0;
          break;
      }

      H264NVENCODER_NVENC_CALL_SAFE(
          encoders_[i]->Reconfigure(&reconfig_params));
    }
  }
}

int32_t H264NvEncoderImpl::Encode(
    const VideoFrame& input_frame,
    const std::vector<VideoFrameType>* frame_types) {
  if (encoders_.empty()) {
    ReportError();
    return WEBRTC_VIDEO_CODEC_UNINITIALIZED;
  }
  if (!encoded_image_callback_) {
    RTC_LOG(LS_WARNING)
        << "Encode() has been called, but a callback function "
           "has not been set with RegisterEncodeCompleteCallback()";
    ReportError();
    return WEBRTC_VIDEO_CODEC_UNINITIALIZED;
  }

  auto const input_buffer = input_frame.video_frame_buffer()->GetI420();

  RTC_DCHECK_EQ(encoders_[0]->GetEncodeWidth(), input_buffer->width());
  RTC_DCHECK_EQ(encoders_[0]->GetEncodeHeight(), input_buffer->height());

  H264NVENCODER_CUDA_CALL(cuCtxPushCurrent(context_));

  bool send_key_frame = key_frame_request_;

  for (size_t i = 0; i < stream_states_.size(); ++i) {
    if (key_frame_request_ && stream_states_[i]) {
      send_key_frame = true;
      break;
    }
  }

  if (!send_key_frame && frame_types) {
    for (size_t i = 0; i < encoders_.size(); ++i) {
      const size_t simulcast_idx = encoders_.size() - 1 - i;
      if (stream_states_[i] && simulcast_idx < frame_types->size() &&
          (*frame_types)[simulcast_idx] == VideoFrameType::kVideoFrameKey) {
        send_key_frame = true;
        break;
      }
    }
  }

  // Encode image for each layer.
  for (size_t i = 0; i < encoders_.size(); ++i) {
    if (!stream_states_[i]) {
      continue;
    }

    VideoFrameType const frame_type = send_key_frame
                                          ? VideoFrameType::kVideoFrameKey
                                          : VideoFrameType::kVideoFrameDelta;

    // EncodeFrame input.
    pictures_[i] = {0};
    pictures_[i].iPicWidth = encoders_[i]->GetEncodeWidth();
    pictures_[i].iPicHeight = encoders_[i]->GetEncodeHeight();
    pictures_[i].iColorFormat = EVideoFormatType::videoFormatI420;
    pictures_[i].uiTimeStamp = input_frame.ntp_time_ms();
    // Downscale images on second and ongoing layers.
    if (i == 0) {
      pictures_[i].iStride[0] = input_buffer->StrideY();
      pictures_[i].iStride[1] = input_buffer->StrideU();
      pictures_[i].iStride[2] = input_buffer->StrideV();
      pictures_[i].pData[0] = const_cast<uint8_t*>(input_buffer->DataY());
      pictures_[i].pData[1] = const_cast<uint8_t*>(input_buffer->DataU());
      pictures_[i].pData[2] = const_cast<uint8_t*>(input_buffer->DataV());
    } else {
      pictures_[i].iStride[0] = downscaled_buffers_[i - 1]->StrideY();
      pictures_[i].iStride[1] = downscaled_buffers_[i - 1]->StrideU();
      pictures_[i].iStride[2] = downscaled_buffers_[i - 1]->StrideV();
      pictures_[i].pData[0] =
          const_cast<uint8_t*>(downscaled_buffers_[i - 1]->DataY());
      pictures_[i].pData[1] =
          const_cast<uint8_t*>(downscaled_buffers_[i - 1]->DataU());
      pictures_[i].pData[2] =
          const_cast<uint8_t*>(downscaled_buffers_[i - 1]->DataV());
      // Scale the image down a number of times by downsampling factor.
      libyuv::I420Scale(
          pictures_[i - 1].pData[0], pictures_[i - 1].iStride[0],
          pictures_[i - 1].pData[1], pictures_[i - 1].iStride[1],
          pictures_[i - 1].pData[2], pictures_[i - 1].iStride[2],
          encoders_[i - 1]->GetEncodeWidth(),
          encoders_[i - 1]->GetEncodeHeight(), pictures_[i].pData[0],
          pictures_[i].iStride[0], pictures_[i].pData[1],
          pictures_[i].iStride[1], pictures_[i].pData[2],
          pictures_[i].iStride[2], encoders_[i]->GetEncodeWidth(),
          encoders_[i]->GetEncodeHeight(), libyuv::kFilterBilinear);
    }

    bool const idr_frame = frame_type == VideoFrameType::kVideoFrameKey;

    auto nvenc_frame = encoders_[i]->GetNextInputFrame();

    H264NVENCODER_NVENC_CALL(encoders_[i]->CopyToDeviceFrame(
        context_,                                              //
        pictures_[i].pData[0],                                 //
        pictures_[i].iStride[0],                               //
        reinterpret_cast<CUdeviceptr>(nvenc_frame->inputPtr),  //
        nvenc_frame->pitch,                                    //
        encoders_[i]->GetEncodeWidth(),                        //
        encoders_[i]->GetEncodeHeight(),
        CU_MEMORYTYPE_HOST,          //
        nvenc_frame->bufferFormat,   //
        nvenc_frame->chromaOffsets,  //
        nvenc_frame->numChromaPlanes));

    NV_ENC_PIC_PARAMS pic_params{NV_ENC_PIC_PARAMS_VER};

    pic_params.frameIdx = input_frame.id();
    pic_params.inputTimeStamp = input_frame.timestamp_us();
    pic_params.encodePicFlags |= NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;

    if (idr_frame) {
      pic_params.encodePicFlags |= NV_ENC_PIC_FLAG_FORCEIDR;
    }

    H264NVENCODER_NVENC_CALL(
        encoders_[i]->EncodeFrame(output_buffer_, &pic_params));

    encoded_images_[i]._frameType = frame_type;
    encoded_images_[i].SetTimestamp(input_frame.timestamp());
    encoded_images_[i].SetSpatialIndex(encoders_.size() - 1 - i);
    encoded_images_[i]._encodedWidth = encoders_[i]->GetEncodeWidth();
    encoded_images_[i]._encodedHeight = encoders_[i]->GetEncodeHeight();

    size_t const encoded_size =
        std::accumulate(output_buffer_.begin(), output_buffer_.end(), 0,
                        [](size_t size, std::vector<uint8_t> const& buffer) {
                          return size + buffer.size();
                        });

    // Encoder can skip frames to save bandwidth in which case
    // |encoded_image_._length| == 0.
    if (encoded_size > 0) {
      encoded_images_[i].SetEncodedData(
          EncodedImageBuffer::Create(encoded_size));
      // Copy the output frame buffers to encoded image data pointer
      std::accumulate(output_buffer_.begin(),  //
                      output_buffer_.end(),    //
                      encoded_images_[i].GetEncodedData()->data(),
                      [](uint8_t* target, std::vector<uint8_t> const& source) {
                        if (source.size() > 0) {
                          memcpy(target, source.data(), source.size());
                        }
                        return target + source.size();
                      });

      // Parse QP.
      h264_bitstream_parser_.ParseBitstream(encoded_images_[i].data(),
                                            encoded_images_[i].size());
      h264_bitstream_parser_.GetLastSliceQp(&encoded_images_[i].qp_);

      // Deliver encoded image.
      CodecSpecificInfo codec_specific;
      codec_specific.codecType = kVideoCodecH264;
      codec_specific.codecSpecific.H264.idr_frame = idr_frame;
      codec_specific.codecSpecific.H264.base_layer_sync = false;
      codec_specific.codecSpecific.H264.temporal_idx = kNoTemporalIdx;
      codec_specific.codecSpecific.H264.packetization_mode =
          packetization_mode_;

      // if (configurations_[i].num_temporal_layers > 1) {
      //   const uint8_t tid = info.sLayerInfo[0].uiTemporalId;
      //   codec_specific.codecSpecific.H264.temporal_idx = tid;
      //   codec_specific.codecSpecific.H264.base_layer_sync =
      //       tid > 0 && tid < tl0sync_limit_[i];
      //   if (codec_specific.codecSpecific.H264.base_layer_sync) {
      //     tl0sync_limit_[i] = tid;
      //   }
      //   if (tid == 0) {
      //     tl0sync_limit_[i] = configurations_[i].num_temporal_layers;
      //   }
      // }

      encoded_image_callback_->OnEncodedImage(encoded_images_[i],
                                              &codec_specific);
    }
  }

  H264NVENCODER_CUDA_CALL(cuCtxPopCurrent(&context_));

  key_frame_request_ = false;

  return WEBRTC_VIDEO_CODEC_OK;
}

VideoEncoder::EncoderInfo H264NvEncoderImpl::GetEncoderInfo() const {
  EncoderInfo info;
  info.supports_native_handle = true;
  info.implementation_name = "NVENC_H264";
  info.scaling_settings =
      VideoEncoder::ScalingSettings(kLowH264QpThreshold, kHighH264QpThreshold);
  info.is_hardware_accelerated = true;
  info.has_internal_source = false;
  info.supports_simulcast = true;
  info.preferred_pixel_formats = {VideoFrameBuffer::Type::kI420};
  return info;
}

void H264NvEncoderImpl::ReportInit() {
  if (has_reported_init_) {
    return;
  }
  RTC_HISTOGRAM_ENUMERATION("WebRTC.Video.H264NvEncoderImpl.Event",
                            kH264EncoderEventInit, kH264EncoderEventMax);
  has_reported_init_ = true;
}

void H264NvEncoderImpl::ReportError() {
  if (has_reported_error_) {
    return;
  }
  RTC_HISTOGRAM_ENUMERATION("WebRTC.Video.H264NvEncoderImpl.Event",
                            kH264EncoderEventError, kH264EncoderEventMax);
  has_reported_error_ = true;
}

void H264NvEncoderImpl::SetStreamState(size_t stream_idx, bool send_stream) {
  if (send_stream && !stream_states_[stream_idx]) {
    // Need a key frame if we have not sent this stream before.
    key_frame_request_ = true;
  }
  stream_states_[stream_idx] = send_stream;
}

}  // namespace webrtc

#undef H264NVENCODER_CUDA_CALL
#undef H264NVENCODER_NVENC_CALL
#undef H264NVENCODER_NVENC_CALL_SAFE

#endif
