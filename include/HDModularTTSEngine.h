#pragma once

#include "MultilingualG2PN.h"

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>
#include <cstdint>

#if defined(_WIN32) || defined(_WIN64)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

// Forward declaration of NCNN net and option
namespace ncnn {
    class Net;
    class Option;
    class Mat;
}

namespace HDModularTTS {

/**
 * MmapBuffer: RAII Zero-Copy Memory Mapped File Wrapper
 * Uses CreateFileMapping on Windows and mmap on Android/Linux.
 */
class MmapBuffer {
public:
    MmapBuffer();
    ~MmapBuffer();

    // Non-copyable
    MmapBuffer(const MmapBuffer&) = delete;
    MmapBuffer& operator=(const MmapBuffer&) = delete;

    // Move-constructible
    MmapBuffer(MmapBuffer&& other) noexcept;
    MmapBuffer& operator=(MmapBuffer&& other) noexcept;

    bool open(const std::string& filepath);
    void close();

    const uint8_t* data() const noexcept { return m_data; }
    size_t size() const noexcept { return m_size; }
    bool isValid() const noexcept { return m_data != nullptr && m_size > 0; }

private:
    const uint8_t* m_data{nullptr};
    size_t m_size{0};

#if defined(_WIN32) || defined(_WIN64)
    HANDLE m_file_handle{INVALID_HANDLE_VALUE};
    HANDLE m_mapping_handle{NULL};
#else
    int m_fd{-1};
#endif
};

/**
 * Mood / Emotion Profiles for Dynamic Expressiveness
 */
enum class Mood {
    NEUTRAL,
    HAPPY,
    SERIOUS,
    STORYTELLING,
    CALM,
    EXCITED
};

struct MoodParams {
    float pitch_shift{1.0f};     // Mean F0 multiplier
    float variance_scale{1.0f};  // Dynamic pitch variance multiplier
    float energy_scale{1.0f};    // Loudness multiplier
    float speed_scale{1.0f};     // Cadence speed multiplier
    std::string name{"neutral"};
};

/**
 * Engine Configuration Options
 */
struct EngineConfig {
    std::string models_dir{"./models"};
    int num_threads{4};
    bool use_vulkan{false};
    int sample_rate{24000};
    int n_fft{1024};
    int hop_length{256};
    float speed_ratio{1.0f};
    int crossfade_samples{512}; // Crossfade between streaming chunks
    Mood mood{Mood::NEUTRAL};   // Dynamic mood/emotion mode
};

/**
 * Engine Performance Benchmarking Telemetry
 */
struct SynthesisStats {
    double time_to_first_audio_ms{0.0};
    double total_synthesis_time_ms{0.0};
    double total_audio_duration_sec{0.0};
    double real_time_factor{0.0}; // total_synthesis_time / total_audio_duration
    size_t total_samples_synthesized{0};
    size_t total_clauses_processed{0};
};

/**
 * HDModularTTSEngine
 * Core C++20 Tencent NCNN Inference Engine.
 * Ultra-Fast Modular Multilingual TTS with Studio HD Vocos Fourier Head.
 */
class HDModularTTSEngine {
public:
    using PCMCallback = std::function<void(const std::vector<float>& pcm_chunk, bool is_final)>;

    explicit HDModularTTSEngine(const EngineConfig& config = EngineConfig{});
    ~HDModularTTSEngine();

    // Initialize models via zero-copy mmap
    bool init();

    // Streaming synthesis: invokes pcm_callback per clause chunk with OLA crossfading
    bool synthesizeStream(const std::string& utf8_text,
                          const std::string& lang_code,
                          PCMCallback pcm_callback);

    // Full one-shot synthesis: returns complete audio buffer
    bool synthesize(const std::string& utf8_text,
                    const std::string& lang_code,
                    std::vector<float>& out_pcm_samples);

    // Telemetry and status
    const SynthesisStats& getStats() const noexcept { return m_stats; }
    const EngineConfig& getConfig() const noexcept { return m_config; }
    bool isReady() const noexcept { return m_is_initialized; }

    // Dynamic Mood / Emotion mode control
    void setMood(Mood mood) noexcept { m_config.mood = mood; }
    Mood getMood() const noexcept { return m_config.mood; }

    // Fast SIMD iSTFT kernel (public for testing or external usage)
    void computeISTFT(const std::vector<float>& mag,
                      const std::vector<float>& phase_x,
                      const std::vector<float>& phase_y,
                      int num_frames,
                      std::vector<float>& out_pcm);

    // Ultra-HD Fourier refinement: Unit-Circle Phase Normalization + Studio Air Boost (7-11.2 kHz)
    static void refineSpectrumSIMD(float* mag, float* phase_x, float* phase_y, int num_bins);

    // Dynamic pitch and prosody shaping per sentence type and mood
    static void applyMoodAndSentenceProsody(std::vector<float>& f0, std::vector<float>& energy,
                                           ClauseType clause, const MoodParams& mood);

    // Mood configuration lookup
    static MoodParams getMoodParams(Mood mood);

    // Natural low-level formant-filtered breath sound for long pauses
    static std::vector<float> generateBreathNoise(int num_samples, float amplitude = 0.004f);

private:
    // Model loading helpers
    bool loadSubnet(const std::string& module_name,
                    std::unique_ptr<ncnn::Net>& net,
                    MmapBuffer& param_mmap,
                    MmapBuffer& bin_mmap);

    // Subnet inference passes
    bool runAcousticEncoder(const std::vector<int32_t>& phoneme_ids,
                            int32_t lang_id,
                            ncnn::Mat& out_text_features);

    bool runDurationPredictor(const ncnn::Mat& text_features,
                              int32_t lang_id,
                              float speed_ratio,
                              ncnn::Mat& out_spec_native);

    bool runVocosBackbone(const ncnn::Mat& spec_features,
                          ncnn::Mat& out_mag,
                          ncnn::Mat& out_phase_x,
                          ncnn::Mat& out_phase_y);

    // Internal iSTFT tables & buffers
    void initISTFT();
    void radix2IFFT(float* real, float* imag, int n);

    EngineConfig m_config;
    SynthesisStats m_stats;
    bool m_is_initialized{false};

    // Linguistic normalizer & router
    MultilingualG2PN m_g2pn;

    // 3 Modular NCNN Networks (Native Female Pretrained Pipeline)
    std::unique_ptr<ncnn::Net> m_net_acoustic;
    std::unique_ptr<ncnn::Net> m_net_duration;
    std::unique_ptr<ncnn::Net> m_net_vocos;

    // Memory-mapped file handles for 3 subnets
    MmapBuffer m_mmap_acoustic_param;
    MmapBuffer m_mmap_acoustic_bin;
    MmapBuffer m_mmap_duration_param;
    MmapBuffer m_mmap_duration_bin;
    MmapBuffer m_mmap_vocos_param;
    MmapBuffer m_mmap_vocos_bin;

    // Precomputed iSTFT tables
    std::vector<float> m_hann_window;
    std::vector<float> m_window_normalization;
    std::vector<float> m_twiddle_cos;
    std::vector<float> m_twiddle_sin;
    std::vector<int>   m_bit_reversal;

    // Streaming state & crossfade buffer
    std::vector<float> m_crossfade_tail;
};

} // namespace HDModularTTS

