// StftProcessor.cpp
#include "StftProcessor.h"
#include <cmath>
#include <algorithm>

static inline float hann512(int n, int N) {
    // Hann with periodic = false (common DSP convention)
    return 0.5f * (1.0f - std::cos(2.0f * float(M_PI) * float(n) / float(N - 1)));
}

StftProcessor::StftProcessor() {
    // windows
    mWin.resize(kNFFT);
    for (int i = 0; i < kNFFT; ++i) mWin[i] = hann512(i, kNFFT);

    // input hop and history
    mHopBuf.assign(kHOP, 0.0f);
    mHist384.assign(384, 0.0f);

    // scratches
    mFFTIn.resize(kNFFT);
    mFFTOut.resize(kNFFT);
    mTime512.assign(kNFFT, 0.0f);
    mTimeWin.assign(kNFFT, 0.0f);

    // OLA ring: power-of-two capacity, plenty of headroom (>= 8 hops)
    const size_t cap = 1u << 15; // 32768 samples
    mOlaBuf.assign(cap, 0.0f);
    mNormBuf.assign(cap, 0.0f);
    mOlaWrite = 0;
    mOlaRead  = 0;
    mOlaMask  = cap - 1;
    mAvail    = 0;
}

void StftProcessor::pushTimeDomain(const float* mono16, int frames) {
    mPushed += static_cast<uint64_t>(frames);
    int idx = 0;
    while (idx < frames) {
        const int need = kHOP - mHopFill;
        const int take = std::min(need, frames - idx);
        std::copy_n(mono16 + idx, take, mHopBuf.begin() + mHopFill);
        mHopFill += take;
        idx      += take;

        if (mHopFill == kHOP) {
            processOneHop();  // consumes mHopBuf
            mHopFill = 0;

            // Update 384-history: drop 96, append 96 new
            // old hist: [0..383] -> keep 288 (from 96..383) and append 96 from hop
            std::move(mHist384.begin() + kHOP, mHist384.end(), mHist384.begin());
            std::copy_n(mHopBuf.begin(), kHOP, mHist384.begin() + (384 - kHOP));
        }
    }
}

void StftProcessor::processOneHop() {
    // Build 512-sample analysis frame:
    // first 32 are zeros, next 480 = 384 from history + 96 new hop
    std::fill(mTime512.begin(), mTime512.end(), 0.0f);
    // copy history (384) to positions [32..415]
    std::copy_n(mHist384.begin(), 384, mTime512.begin() + 32);
    // copy new hop (96) to positions [416..511]
    std::copy_n(mHopBuf.begin(), kHOP, mTime512.begin() + 32 + 384);

    // Analysis window
    for (int i = 0; i < kNFFT; ++i) mTimeWin[i] = mTime512[i] * mWin[i];

    // Pack to complex
    for (int i = 0; i < kNFFT; ++i) mFFTIn[i] = std::complex<float>(mTimeWin[i], 0.0f);

    // FFT
    mFFTOut = mFFTIn;
    fft(mFFTOut, /*inverse=*/false);

    // Identity processing (Y = X)
    // (do nothing)

    // iFFT
    auto y = mFFTOut;
    fft(y, /*inverse=*/true); // returns scaled by 1/N internally

    // Synthesis window and OLA
    for (int i = 0; i < kNFFT; ++i) mTimeWin[i] = y[i].real() * mWin[i];
    olaAdd(mTimeWin.data());

    // After OLA add, we made exactly kHOP new samples available.
    mAvail += kHOP;
    mHops += 1;
}

int StftProcessor::popTimeDomain(float* out16, int maxFrames) {
    const int want = std::min<int>(maxFrames, static_cast<int>(mAvail));
    for (int i = 0; i < want; ++i) {
        const size_t idx = (mOlaRead + i) & mOlaMask;
        const float n = mNormBuf[idx];
        out16[i] = (n > kEps) ? (mOlaBuf[idx] / n) : 0.0f;

        // clear after reading (keeps buffers bounded)
        mOlaBuf[idx]  = 0.0f;
        mNormBuf[idx] = 0.0f;
    }
    mOlaRead = (mOlaRead + want) & mOlaMask;
    mAvail  -= want;
    mPopped += static_cast<uint64_t>(want);
    return want;
}

// ===== FFT (radix-2, N=512) =====
void StftProcessor::fft(std::vector<std::complex<float>>& a, bool inverse) {
    // length must be power of two (512)
    const size_t n = a.size();

    // bit-reverse
    size_t j = 0;
    for (size_t i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    // Cooley–Tukey
    for (size_t len = 2; len <= n; len <<= 1) {
        const float ang = (inverse ? 2.0f : -2.0f) * float(M_PI) / float(len);
        const std::complex<float> wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            const size_t half = len >> 1;
            for (size_t k = 0; k < half; ++k) {
                auto u = a[i + k];
                auto v = a[i + k + half] * w;
                a[i + k]         = u + v;
                a[i + k + half]  = u - v;
                w *= wlen;
            }
        }
    }
    // scale for iFFT
    if (inverse) {
        const float invN = 1.0f / float(n);
        for (auto& z : a) z *= invN;
    }
}

void StftProcessor::olaAdd(const float* block512) {
    for (int i = 0; i < kNFFT; ++i) {
        const size_t idx = (mOlaWrite + i) & mOlaMask;
        mOlaBuf[idx]  += block512[i];
        mNormBuf[idx] += mWin[i] * mWin[i];
    }
    mOlaWrite = (mOlaWrite + kHOP) & mOlaMask;
}