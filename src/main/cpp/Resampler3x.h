#pragma once
#include <vector>
#include <cstddef>
#include <algorithm>

class Resampler3x {
public:
    enum class Mode { DownBy3, UpBy3 };

    explicit Resampler3x(Mode m = Mode::DownBy3) : mMode(m) {}

    void setMode(Mode m) { mMode = m; reset(); }
    void reset() { mPrevSample = 0.0f; mHadPrev = false; }

    // Returns number of output frames produced.
    // For DownBy3: requires inFrames multiple of 3 (we use 96 -> 32).
    // For UpBy3: produces exactly 3*outFrames, using simple linear interpolation.
    int process(const float* in, int inFrames, float* out, int outMaxFrames) {
        return (mMode == Mode::DownBy3)
               ? processDown3(in, inFrames, out, outMaxFrames)
               : processUp3(in, inFrames, out, outMaxFrames);
    }

private:
    Mode  mMode;
    float mPrevSample = 0.0f;
    bool  mHadPrev = false;

    int processDown3(const float* in, int inFrames, float* out, int outMaxFrames) {
        const int groups = inFrames / 3;
        const int produced = std::min(groups, outMaxFrames);
        for (int g = 0; g < produced; ++g) {
            const float s0 = in[g*3 + 0];
            const float s1 = in[g*3 + 1];
            const float s2 = in[g*3 + 2];
            out[g] = (s0 + s1 + s2) * (1.0f/3.0f);
        }
        return produced;
    }

    int processUp3(const float* in, int inFrames, float* out, int outMaxFrames) {
        if (inFrames <= 0) return 0;
        const int need = inFrames * 3; // EXACT 3x
        const int producedMax = std::min(need, outMaxFrames);
        int outIdx = 0;

        // For each input sample i, interpolate towards the next (or hold at end)
        for (int i = 0; i < inFrames && (outIdx + 3) <= producedMax; ++i) {
            const float x0 = in[i];
            const float x1 = (i + 1 < inFrames) ? in[i + 1] : x0; // hold last
            const float d  = (x1 - x0) * (1.0f / 3.0f);
            out[outIdx++] = x0;            // 0/3
            out[outIdx++] = x0 + d;        // 1/3
            out[outIdx++] = x0 + 2.0f*d;   // 2/3
        }

        // Save tail for potential continuity use later
        mPrevSample = in[inFrames - 1];
        mHadPrev = true;
        return outIdx; // == 3 * inFrames unless clipped
    }

};