// StftProcessor.h
#pragma once
#include <vector>
#include <complex>
#include <cstddef>

class StftProcessor {
public:
    // NFFT=512, hop=96, analysis frame=480 (zero-pad to 512)
    StftProcessor();

    // Feed mono@16k time-domain samples (any count). Internally,
    // every 96 samples it makes a 480-frame with 384 overlap, pads to 512,
    // FFT -> identity -> iFFT -> OLA into an internal FIFO.
    void pushTimeDomain(const float* mono16, int frames);

    // Pop up to maxFrames mono@16k samples produced by OLA (normalized).
    // Returns frames actually written to out16.
    int popTimeDomain(float* out16, int maxFrames);

    uint64_t framesPushed() const { return mPushed; }
    uint64_t framesPopped() const { return mPopped; }
    uint64_t hopsProcessed() const { return mHops; }

private:
    // --- constants ---
    static constexpr int kNFFT   = 512;
    static constexpr int kHOP    = 96;
    static constexpr int kFRAME  = 480; // 384 overlap + 96 new
    static constexpr float kEps  = 1e-8f;

    // --- analysis/synthesis window ---
    std::vector<float> mWin;        // Hann(512)

    // --- small input staging (collect hops of 96) ---
    std::vector<float> mHopBuf;     // size kHOP
    int                mHopFill = 0;

    // --- rolling history for 384 overlap ---
    std::vector<float> mHist384;    // size 384

    // --- scratch buffers for one STFT block ---
    std::vector<std::complex<float>> mFFTIn;   // 512
    std::vector<std::complex<float>> mFFTOut;  // 512
    std::vector<float>               mTime512; // 512
    std::vector<float>               mTimeWin; // 512

    // --- OLA FIFO (circular) + normalization FIFO (sum of win^2) ---
    std::vector<float> mOlaBuf;   // big ring
    std::vector<float> mNormBuf;  // big ring
    size_t             mOlaWrite = 0;
    size_t             mOlaRead  = 0;
    size_t             mOlaMask  = 0; // capacity-1 (power of two)
    size_t             mAvail    = 0; // frames available to pop

    uint64_t mPushed = 0;
    uint64_t mPopped = 0;
    uint64_t mHops   = 0;

    // --- FFT helpers (radix-2 iterative, N=512) ---
    void fft(std::vector<std::complex<float>>& a, bool inverse);

    // One complete STFT frame (using the 96 samples currently in mHopBuf).
    void processOneHop();

    // push to OLA ring (timeWin added, norm adds win^2)
    void olaAdd(const float* block512);
};