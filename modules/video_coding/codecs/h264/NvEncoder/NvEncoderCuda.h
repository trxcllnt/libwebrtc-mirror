/*
 * Copyright 2017-2020 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#pragma once

#ifndef MODULES_VIDEO_CODING_CODECS_H264_NVENCODER_NVENCODERCUDA_H_
#define MODULES_VIDEO_CODING_CODECS_H264_NVENCODER_NVENCODERCUDA_H_

// Everything declared in this header is only required when WebRTC is
// build with NVENC H264 support, please do not move anything out of the
// #ifdef unless needed and tested.
#ifdef WEBRTC_USE_H264_NVENC

#include "modules/video_coding/codecs/h264/NvEncoder/NvEncoder.h"

#include <cuda.h>
#include <stdint.h>
#include <mutex>
#include <vector>

#define CUDA_DRVAPI_CALL(expr)                             \
  do {                                                     \
    CUresult result = (expr);                              \
    if (result != CUDA_SUCCESS) {                          \
      const char* name = NULL;                             \
      const char* error = NULL;                            \
      cuGetErrorName(result, &name);                       \
      cuGetErrorString(result, &error);                    \
      std::ostringstream errorLog;                         \
      errorLog << "\"" << #expr << "\"\n"                  \
               << " returned CUDA error " << name << ":\n" \
               << error;                                   \
      NVENC_THROW_ERROR(errorLog.str(), result);           \
    }                                                      \
  } while (0)

/**
 *  @brief Encoder for CUDA device memory.
 */
class NvEncoderCuda : public NvEncoder {
 public:
  NvEncoderCuda();
  virtual ~NvEncoderCuda();

  int32_t Initialize(CUcontext cuContext,
                     uint32_t nWidth,
                     uint32_t nHeight,
                     NV_ENC_BUFFER_FORMAT eBufferFormat,
                     uint32_t nExtraOutputDelay = 3,
                     bool bMotionEstimationOnly = false,
                     bool bOPInVideoMemory = false);

  /**
   *  @brief This is a static function to copy input data from host memory to
   * device memory. This function assumes YUV plane is a single contiguous
   * memory segment.
   */
  static int32_t CopyToDeviceFrame(CUcontext device,
                                   void* pSrcFrame,
                                   uint32_t nSrcPitch,
                                   CUdeviceptr pDstFrame,
                                   uint32_t dstPitch,
                                   int width,
                                   int height,
                                   CUmemorytype srcMemoryType,
                                   NV_ENC_BUFFER_FORMAT pixelFormat,
                                   const uint32_t dstChromaOffsets[],
                                   uint32_t numChromaPlanes,
                                   bool bUnAlignedDeviceCopy = false,
                                   CUstream stream = NULL);

  /**
   *  @brief This is a static function to copy input data from host memory to
   * device memory. Application must pass a seperate device pointer for each YUV
   * plane.
   */
  static int32_t CopyToDeviceFrame(CUcontext device,
                                   void* pSrcFrame,
                                   uint32_t nSrcPitch,
                                   CUdeviceptr pDstFrame,
                                   uint32_t dstPitch,
                                   int width,
                                   int height,
                                   CUmemorytype srcMemoryType,
                                   NV_ENC_BUFFER_FORMAT pixelFormat,
                                   CUdeviceptr dstChromaPtr[],
                                   uint32_t dstChromaPitch,
                                   uint32_t numChromaPlanes,
                                   bool bUnAlignedDeviceCopy = false);

  /**
   *  @brief This function sets input and output CUDA streams
   */
  int32_t SetIOCudaStreams(NV_ENC_CUSTREAM_PTR inputStream,
                           NV_ENC_CUSTREAM_PTR outputStream);

 protected:
  /**
   *  @brief This function is used to release the input buffers allocated for
   * encoding. This function is an override of virtual function
   * NvEncoder::ReleaseInputBuffers().
   */
  virtual void ReleaseInputBuffers() override;

 private:
  /**
   *  @brief This function is used to allocate input buffers for encoding.
   *  This function is an override of virtual function
   * NvEncoder::AllocateInputBuffers().
   */
  virtual int32_t AllocateInputBuffers(int32_t numInputBuffers) override;

 private:
  /**
   *  @brief This is a private function to release CUDA device memory used for
   * encoding.
   */
  void ReleaseCudaResources();

 protected:
  CUcontext m_cuContext;

 private:
  size_t m_cudaPitch = 0;
};

#endif
#endif
