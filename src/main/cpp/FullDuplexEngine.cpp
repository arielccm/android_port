#include "FullDuplexEngine.h"
#include <logging_macros.h> // same macro set used in the sample
#include <cinttypes>
#include <atomic>
#include <chrono>
#ifdef __ANDROID__
#include <sys/resource.h>
#endif

// --- Small helpers for (de)interleaving ---
static inline void deinterleaveStereo(const float* inter, int frames,
                                      float* L, float* R) {
    const float* p = inter;
    for (int i = 0; i < frames; ++i) {
        L[i] = *p++;
        R[i] = *p++;
    }
}
static inline void interleaveStereo(const float* L, const float* R, int frames,
                                    float* inter) {
    float* p = inter;
    for (int i = 0; i < frames; ++i) {
        *p++ = L[i];
        *p++ = R[i];
    }
}
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

    // Prime output ring with a few bursts of silence so the first callbacks do not underflow.
    {
        const int kPrimeBursts = 20; // ~20 * 96 frames @48k ≈ 40 ms of audio
        std::vector<float> zeros(static_cast<size_t>(fpb) * ch, 0.0f);
        for (int i = 0; i < kPrimeBursts; ++i) {
            // Ignore return; if the ring can't take more it will just stop filling.
            (void)mOutRing.writeInterleaved(zeros.data(), fpb);
        }
    }
// Record start time (optional future use: grace period for counters)
    mStartTime = std::chrono::steady_clock::now();

    // NEW: per-channel scratch
    mL48.resize(fpb);
    mR48.resize(fpb);
    mL16.resize(fpb / 3);
    mR16.resize(fpb / 3);
    mL48b.resize(fpb * 3);
    mR48b.resize(fpb * 3);
    mTmpOut.resize(static_cast<size_t>(fpb) * 3 * ch);

    // NEW (M3): mono buffers
    mMono16.resize(fpb / 3);
    mBlkMono16.resize(fpb / 3);
    mUp48Mono.resize(fpb * 3);
    // STFT hop buffers (one hop = 96 @16k)
    mHopIn16.resize(96);
    mHopOut16.resize(96);

    mBlkL16.resize(fpb / 3);
    mBlkR16.resize(fpb / 3);

    const int32_t cap16 = (sr / 5) / 3; // 48k/5/3 ≈ 3200
    if (!mMid16kL.init(cap16, 1)) return false;
    if (!mMid16kR.init(cap16, 1)) return false;
    if (!mMid16kMono.init(cap16, 1)) return false;  // NEW

// Reset resamplers (not strictly necessary, but tidy)
    mDownL.reset(); mDownR.reset();
    mUpL.reset();   mUpR.reset();
    mUpMono.reset(); // NEW
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
    #ifdef __ANDROID__
    setpriority(PRIO_PROCESS, 0, -18);
    #endif
    auto lastLog = std::chrono::steady_clock::now();

    while (mRunning.load(std::memory_order_acquire)) {
        // 1) BLOCKING READ from input
        oboe::ResultWithValue<int32_t> res =
                mIn->read(mTmpIn.data(), fpb, 10 * 1000 * 1000 /* 10ms timeout */);

        if (!res) {
            continue; // glitchgit
        }

        int32_t got = res.value();
        if (got <= 0) continue;

        // 2) push to input ring
        int32_t wrote = mInRing.writeInterleaved(mTmpIn.data(), got);
        if (wrote < got) mOverflows.fetch_add(got - wrote);

        // 3) 48k -> 16k -> (mono) -> 48k round-trip
        int32_t canXfer = std::min(mInRing.availableToRead(), mOutRing.availableToWrite());
        while (canXfer >= fpb) {
            // read one burst @48k interleaved
            int32_t rd = mInRing.readInterleaved(mTmpXfer.data(), fpb);
            if (rd == fpb) {
                // deinterleave to L/R @48k
                deinterleaveStereo(mTmpXfer.data(), fpb, mL48.data(), mR48.data());

                // downsample by 3 -> 16k (expect fpb/3 frames)
                const int out16L = mDownL.process(mL48.data(), fpb, mL16.data(), (int)mL16.size());
                const int out16R = mDownR.process(mR48.data(), fpb, mR16.data(), (int)mR16.size());
                const int out16  = std::min(out16L, out16R);

                // --- Milestone 3: mix to mono @16k
                for (int i = 0; i < out16; ++i) {
                    mMono16[i] = 0.5f * (mL16[i] + mR16[i]);
                }

                // write mono to mono ring (decoupling point for future STFT/model)
                int wM = mMid16kMono.writeInterleaved(mMono16.data(), out16);
                if (wM < out16) {
                    mOverflows.fetch_add(out16 - wM);
                }

// Feed STFT in 96-sample hops, pop 96 back each hop, upsample to 48k, duplicate to stereo
                while (mMid16kMono.availableToRead() >= 96) {
                    (void)mMid16kMono.readInterleaved(mHopIn16.data(), 96);

                    // push 96 into STFT
                    mStft.pushTimeDomain(mHopIn16.data(), 96);

                    // pop exactly 96 out of STFT
                    const int got16 = mStft.popTimeDomain(mHopOut16.data(), 96);
                    if (got16 == 96) {
                        // upsample 96 -> 288 @48k
                        const int up = mUpMono.process(mHopOut16.data(), 96, mUp48Mono.data(), (int)mUp48Mono.size());
                        const int upFrames = up; // allow full 288 frames from one hop

                        // duplicate mono to stereo
                        for (int i = 0; i < upFrames; ++i) {
                            mL48b[i] = mUp48Mono[i];
                            mR48b[i] = mUp48Mono[i];
                        }

                        // interleave and write to out ring
                        interleaveStereo(mL48b.data(), mR48b.data(), upFrames, mTmpOut.data());
                        int32_t wr = mOutRing.writeInterleaved(mTmpOut.data(), upFrames);
                        if (wr < upFrames) mOverflows.fetch_add(upFrames - wr);
                    }
                }
            }
            canXfer = std::min(mInRing.availableToRead(), mOutRing.availableToWrite());
        }

        // --- Periodic stats log every 1s ---
        auto now = std::chrono::steady_clock::now();
        if (now - lastLog > std::chrono::seconds(1)) {
            lastLog = now;
            // STFT counters
            uint64_t hops   = mStft.hopsProcessed();
            uint64_t pushed = mStft.framesPushed();
            uint64_t popped = mStft.framesPopped();

            LOGD("Stats: InRing=%d OutRing=%d Overflows=%" PRId64 " Underflows=%" PRId64
                         " | STFT hops +%llu (tot %llu), push +%llu, pop +%llu",
                 mInRing.availableToRead(),
                 mOutRing.availableToRead(),
                 static_cast<int64_t>(mOverflows.load()),
                 static_cast<int64_t>(mUnderflows.load()),
                 (unsigned long long)(hops   - mDbgLastHops),
                 (unsigned long long)hops,
                 (unsigned long long)(pushed - mDbgLastPushed),
                 (unsigned long long)(popped - mDbgLastPopped));

            mDbgLastHops   = hops;
            mDbgLastPushed = pushed;
            mDbgLastPopped = popped;
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
        // During warm-up (first ~300 ms after start), do not count underflows
        auto now = std::chrono::steady_clock::now();
        bool warming = (now - mStartTime) < std::chrono::milliseconds(300);
        if (!warming) {
            mUnderflows.fetch_add((numFrames - total));
        }
    }
    return numFrames; // we always fill the buffer handed to the callback
}