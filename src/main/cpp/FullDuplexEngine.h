#pragma once
#include <memory>
#include <thread>
#include <atomic>
#include <vector>
#include <oboe/Oboe.h>
#include "RingBuffer.h"

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

    std::thread mThread;
    std::atomic<bool> mRunning{false};

    // Scratch buffers sized to framesPerBurst * channels (resized on start)
    std::vector<float> mTmpIn;
    std::vector<float> mTmpXfer;

    // For simple stats (optional)
    std::atomic<int64_t> mUnderflows{0};
    std::atomic<int64_t> mOverflows{0};
};
