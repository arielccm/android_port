#include "FullDuplexEngine.h"
#include <logging_macros.h> // same macro set used in the sample
#include <cinttypes>
#include <atomic>
#include <chrono>
bool FullDuplexEngine::start() {
    if (!mIn || !mOut) return false;

    const int32_t ch = mOut->getChannelCount();
    const int32_t fpb = mOut->getFramesPerBurst();
    const int32_t sr  = mOut->getSampleRate();
    (void)sr; // not used here, but useful to log if you want

    // ~200 ms of capacity is a nice safety margin but still low-latency
    const int32_t capFrames = sr / 5; // e.g., 48000/5 = 9600
    if (!mInRing.init(capFrames, ch))  return false;
    if (!mOutRing.init(capFrames, ch)) return false;

    mTmpIn.resize(static_cast<size_t>(fpb) * ch);
    mTmpXfer.resize(static_cast<size_t>(fpb) * ch);

    // Start streams so read()/callback are active
    {
        auto rIn = mIn->requestStart();
        LOGI("FullDuplexEngine.start(): requestStart(input) -> %s", oboe::convertToText(rIn));
        if (rIn != oboe::Result::OK) {
            return false;
        }
        auto rOut = mOut->requestStart();
        LOGI("FullDuplexEngine.start(): requestStart(output) -> %s", oboe::convertToText(rOut));
        if (rOut != oboe::Result::OK) {
            (void)mIn->requestStop(); // best-effort rollback
            return false;
        }
    }

    mRunning.store(true, std::memory_order_release);
    mThread = std::thread(&FullDuplexEngine::ioThreadFunc, this);
    return true;
}

void FullDuplexEngine::stop() {
    if (mRunning.exchange(false)) {
        if (mThread.joinable()) mThread.join();
    }

    // Stop streams (best effort)
    if (mOut) {
        auto r = mOut->requestStop();
        if (r != oboe::Result::OK) {
            LOGW("FullDuplexEngine.stop(): requestStop(output) -> %s", oboe::convertToText(r));
        }
    }
    if (mIn) {
        auto r = mIn->requestStop();
        if (r != oboe::Result::OK) {
            LOGW("FullDuplexEngine.stop(): requestStop(input) -> %s", oboe::convertToText(r));
        }
    }
}

void FullDuplexEngine::ioThreadFunc() {
    const int32_t fpb = mOut->getFramesPerBurst();

    auto lastLog = std::chrono::steady_clock::now();

    while (mRunning.load(std::memory_order_acquire)) {
        // 1) BLOCKING READ from input
        oboe::ResultWithValue<int32_t> res =
                mIn->read(mTmpIn.data(), fpb, 20 * 1000 * 1000 /* 20ms timeout */);

        if (!res) {
            continue; // glitch
        }

        int32_t got = res.value();
        if (got <= 0) continue;

        // 2) push to input ring
        int32_t wrote = mInRing.writeInterleaved(mTmpIn.data(), got);
        if (wrote < got) mOverflows.fetch_add(got - wrote);

        // 3) passthrough
        int32_t canXfer = std::min(mInRing.availableToRead(), mOutRing.availableToWrite());
        while (canXfer >= fpb) {
            int32_t rd = mInRing.readInterleaved(mTmpXfer.data(), fpb);
            if (rd == fpb) {
                int32_t wr = mOutRing.writeInterleaved(mTmpXfer.data(), fpb);
                if (wr < fpb) mOverflows.fetch_add(fpb - wr);
            }
            canXfer = std::min(mInRing.availableToRead(), mOutRing.availableToWrite());
        }

        // --- Periodic stats log every 1s ---
        auto now = std::chrono::steady_clock::now();
        if (now - lastLog > std::chrono::seconds(1)) {
            lastLog = now;
            LOGD("Stats: InRing=%d OutRing=%d Overflows=%" PRId64 " Underflows=%" PRId64,
                 mInRing.availableToRead(),
                 mOutRing.availableToRead(),
                 static_cast<int64_t>(mOverflows.load()),
                 static_cast<int64_t>(mUnderflows.load()));
        }
    }
}

int32_t FullDuplexEngine::pullTo(float* out, int32_t numFrames) {
    int32_t total = 0;
    while (total < numFrames) {
        int32_t got = mOutRing.readInterleaved(out + (static_cast<size_t>(total) * mOut->getChannelCount()),
                                               numFrames - total);
        if (got <= 0) break;
        total += got;
    }
    if (total < numFrames) {
        // Underflow: zero-fill the rest so we never hand garbage to the device
        const int32_t ch = mOut->getChannelCount();
        std::memset(out + static_cast<size_t>(total) * ch, 0,
                    static_cast<size_t>(numFrames - total) * ch * sizeof(float));
        mUnderflows.fetch_add((numFrames - total));
    }
    return numFrames; // we always fill the buffer handed to the callback
}