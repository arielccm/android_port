#pragma once
#include <atomic>
#include <cstdint>
#include <vector>
#include <cstring>
#include <algorithm>

/**
 * Lock-free SPSC ring buffer for interleaved float audio.
 * Indices are in FRAMES; buffer stores interleaved floats.
 */
class RingBuffer {
public:
    RingBuffer() = default;

    // capacityFrames: how many frames (each frame = 'channels' samples)
    bool init(int32_t capacityFrames, int32_t channels) {
        if (capacityFrames <= 0 || channels <= 0) return false;
        mChannels = channels;
        mCapacityFrames = nextPow2(capacityFrames);   // power-of-two simplifies wrap
        mMask = mCapacityFrames - 1;
        mData.resize(static_cast<size_t>(mCapacityFrames) * mChannels);
        mRead.store(0, std::memory_order_release);
        mWrite.store(0, std::memory_order_release);
        return true;
    }

    int32_t channels() const { return mChannels; }
    int32_t capacityFrames() const { return mCapacityFrames; }

    // Frames currently available to READ
    int32_t availableToRead() const {
        uint64_t r = mRead.load(std::memory_order_acquire);
        uint64_t w = mWrite.load(std::memory_order_acquire);
        return static_cast<int32_t>(w - r);
    }

    // Free frames available for WRITE
    int32_t availableToWrite() const {
        return mCapacityFrames - availableToRead();
    }

    // Write up to 'frames' interleaved frames. Returns frames actually written.
    int32_t writeInterleaved(const float* src, int32_t frames) {
        frames = std::max<int32_t>(0, std::min(frames, availableToWrite()));
        if (frames == 0) return 0;

        uint64_t w = mWrite.load(std::memory_order_relaxed);
        int32_t first = std::min<int32_t>(frames, mCapacityFrames - static_cast<int32_t>(w & mMask));
        int32_t second = frames - first;

        float* dstA = &mData[(static_cast<size_t>(w & mMask) * mChannels)];
        std::memcpy(dstA, src, static_cast<size_t>(first) * mChannels * sizeof(float));

        if (second > 0) {
            const float* srcB = src + static_cast<size_t>(first) * mChannels;
            float* dstB = mData.data(); // wrap
            std::memcpy(dstB, srcB, static_cast<size_t>(second) * mChannels * sizeof(float));
        }

        mWrite.store(w + frames, std::memory_order_release);
        return frames;
    }

    // Read up to 'frames' interleaved frames. Returns frames actually read.
    int32_t readInterleaved(float* dst, int32_t frames) {
        frames = std::max<int32_t>(0, std::min(frames, availableToRead()));
        if (frames == 0) return 0;

        uint64_t r = mRead.load(std::memory_order_relaxed);
        int32_t first = std::min<int32_t>(frames, mCapacityFrames - static_cast<int32_t>(r & mMask));
        int32_t second = frames - first;

        const float* srcA = &mData[(static_cast<size_t>(r & mMask) * mChannels)];
        std::memcpy(dst, srcA, static_cast<size_t>(first) * mChannels * sizeof(float));

        if (second > 0) {
            const float* srcB = mData.data(); // wrap
            float* dstB = dst + static_cast<size_t>(first) * mChannels;
            std::memcpy(dstB, srcB, static_cast<size_t>(second) * mChannels * sizeof(float));
        }

        mRead.store(r + frames, std::memory_order_release);
        return frames;
    }

private:
    static int32_t nextPow2(int32_t v) {
        v--;
        v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
        v++;
        return v < 2 ? 2 : v;
    }

    std::vector<float> mData;
    int32_t mChannels{1};
    int32_t mCapacityFrames{0};
    int32_t mMask{0};

    std::atomic<uint64_t> mRead{0};   // in FRAMES
    std::atomic<uint64_t> mWrite{0};  // in FRAMES
};