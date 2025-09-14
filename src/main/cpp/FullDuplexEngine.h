#pragma once
#include <memory>
#include <thread>
#include <atomic>
#include <vector>
#include <oboe/Oboe.h>
#include "RingBuffer.h"
#include "Resampler3x.h"
#include <chrono>
#include "StftProcessor.h"

class FullDuplexEngine {
public:
    FullDuplexEngine() = default;
    ~FullDuplexEngine() { stop(); }

    void setSharedInputStream(const std::shared_ptr<oboe::AudioStream>& in)  { mIn = in; }
    void setSharedOutputStream(const std::shared_ptr<oboe::AudioStream>& out){ mOut = out; }

    bool start();
    void stop();

    // Called from Oboe playback callback to pull audio for output
    int32_t pullTo(float* out, int32_t numFrames);

private:
    void ioThreadFunc();

    std::shared_ptr<oboe::AudioStream> mIn;
    std::shared_ptr<oboe::AudioStream> mOut;

    RingBuffer mInRing;      // 48k stereo input queue
    RingBuffer mOutRing;     // 48k stereo output queue

    // NEW: mid-rate mono rings per channel (16 kHz)
    RingBuffer mMid16kL;
    RingBuffer mMid16kR;

    // NEW: resamplers
    Resampler3x mDownL{Resampler3x::Mode::DownBy3};
    Resampler3x mDownR{Resampler3x::Mode::DownBy3};
    Resampler3x mUpL  {Resampler3x::Mode::UpBy3};
    Resampler3x mUpR  {Resampler3x::Mode::UpBy3};

    // NEW step 3: mono 16 kHz ring and buffers
    RingBuffer mMid16kMono;        // 16 kHz mono queue

    std::vector<float> mMono16;    // mixed L/R -> mono @16k for current chunk (size fpb/3)
    std::vector<float> mBlkMono16; // temp pull from mono ring @16k (size fpb/3)
    std::vector<float> mUp48Mono;  // upsampled mono @48k (size fpb)
    Resampler3x mUpMono{Resampler3x::Mode::UpBy3};

    std::thread mThread;
    std::atomic<bool> mRunning{false};

    // Scratch buffers sized to framesPerBurst * channels (resized on start)
    std::vector<float> mTmpIn;      // interleaved @48k, size fpb*ch
    std::vector<float> mTmpXfer;    // interleaved @48k, size fpb*ch
    std::vector<float> mL48, mR48;  // deinterleaved @48k, size fpb
    std::vector<float> mL16, mR16;  // @16k, size fpb/3
    std::vector<float> mL48b, mR48b;// upsampled back to @48k, size fpb
    std::vector<float> mTmpOut;     // interleaved @48k, size fpb*ch
    // NEW: steady 16k chunk buffers (no allocs in loop)
    std::vector<float> mBlkL16; // reused 16k chunk (left or mono)
    std::vector<float> mBlkR16; // reused 16k chunk (right)

    // STFT processor @16k mono
    StftProcessor mStft;
// 16k hop buffers (exactly one hop = 96)
    std::vector<float> mHopIn16;   // size 96
    std::vector<float> mHopOut16;  // size 96

    // For simple stats (optional)
    std::atomic<int64_t> mUnderflows{0};
    std::atomic<int64_t> mOverflows{0};
    // Debug: STFT counters snapshot for logging
    uint64_t mDbgLastHops{0};
    uint64_t mDbgLastPushed{0};
    uint64_t mDbgLastPopped{0};
    std::chrono::steady_clock::time_point mStartTime{};
};
