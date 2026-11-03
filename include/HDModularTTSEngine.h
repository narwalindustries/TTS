#pragma once

#include "MultilingualG2PN.h"
#include "ProsodyController.h"

#include <string>
#include <vector>
#include <array>
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

// Forward declaration of NCNN net, option, and mat
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
 * Engine Configuration Options
 */
struct EngineConfig {
    std::string models_dir{"./models"};
    std::string speaker_bin_path{"./models/universal_speaker.bin"};
    int num_threads{4};
    bool use_vulkan{false};
    int sample_rate{24000};
    int n_fft{1024};
    int hop_length{256};
    float speed_ratio{1.0f};
    int crossfade_samples{512}; // 512-sample OLA streaming crossfade
    Mood mood{Mood::NEUTRAL};   // Dynamic C++20 mood/emotion mode
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
 * Core C++20 Tencent NCNN Inference Engine implementing the 8-Step IMS-Toucan Architecture:
 *   Step 1 & 2: MultilingualG2PN 21-Language Colloquial Slang & Native Accent Mapper
 *   Step 3:     15.5 KB constexpr 62-D Articulatory Phoneme Lookup Table [N, 62]
 *   Step 4:     Subnet 1 (`toucan_encoder.bin`) -> `encoder_hidden [N, 512]`
 *   Step 5A:    Subnet 2A (`explicit_prosody.bin`) -> `durations [N]`, `f0_pitch [N]`, `energy [N]`
 *   Step 5B:    Native C++20 `ProsodyController` (Mood + Sentence Intonation + Tonal Safety)
 *   Step 5C:    Subnet 2B (`toucan_cfm_decoder.bin`) -> `[T, 80]` 24kHz Mel Spectrogram
 *   Step 6:     Studio HD Vocos (`vocos_hd_backbone.bin`) -> Magnitude & Phase
 *   Step 7:     1024-pt Hann Cooley-Tukey Radix-2 iSTFT + Unit-Circle Phase Norm + +1.5dB Air-Band Boost
 *   Step 8:     512-Sample Pitch-Aligned OLA Streaming Crossfade
 */
class HDModularTTSEngine {
public:
    static constexpr size_t SPEAKER_EMBED_DIM = 192;
    static constexpr size_t SPEAKER_BIN_BYTES = SPEAKER_EMBED_DIM * sizeof(float); // 768 bytes
    static constexpr size_t ENCODER_HIDDEN_DIM = 512;
    static constexpr size_t MEL_BANDS = 80;

    using PCMCallback = std::function<void(const std::vector<float>& pcm_chunk, bool is_final)>;

    explicit HDModularTTSEngine(const EngineConfig& config = EngineConfig{});
    ~HDModularTTSEngine();

    // Initialize models & 768-byte universal_speaker.bin via zero-copy mmap
    bool init();

    // Streaming synthesis: invokes pcm_callback per clause chunk with 512-sample OLA crossfading
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

    // Access locked 192-D ECAPA-TDNN speaker timbre vector (768 bytes)
    const std::array<float, SPEAKER_EMBED_DIM>& getUniversalSpeakerVector() const noexcept {
        return m_universal_speaker;
    }

    // Dynamic Mood / Emotion mode control
    void setMood(Mood mood) noexcept { m_config.mood = mood; }
    Mood getMood() const noexcept { return m_config.mood; }

    // Step 4: Subnet 1 (`toucan_encoder.bin`)
    bool runToucanEncoder(const std::vector<float>& articulatory_features,
                          int num_tokens,
                          int32_t lang_id,
                          ncnn::Mat& out_encoder_hidden);

    // Step 5A: Subnet 2A (`explicit_prosody.bin`)
    bool runExplicitProsody(const ncnn::Mat& encoder_hidden,
                            ncnn::Mat& out_durations,
                            ncnn::Mat& out_f0_pitch,
                            ncnn::Mat& out_energy);

    // Step 5C: Subnet 2B (`toucan_cfm_decoder.bin`)
    bool runToucanCFMDecoder(const ncnn::Mat& encoder_hidden,
                             const std::vector<float>& durations,
                             const std::vector<float>& f0_pitch,
                             const std::vector<float>& energy,
                             ncnn::Mat& out_mel_spec);

    // Step 6: Studio HD Vocos (`vocos_hd_backbone.bin`)
    bool runVocosBackbone(const ncnn::Mat& spec_features,
                          ncnn::Mat& out_mag,
                          ncnn::Mat& out_phase_x,
                          ncnn::Mat& out_phase_y);

    // Step 7: Fast SIMD iSTFT kernel (1024-point Hann Cooley-Tukey Radix-2 iSTFT at 24,000 Hz)
    void computeISTFT(const std::vector<float>& mag,
                      const std::vector<float>& phase_x,
                      const std::vector<float>& phase_y,
                      int num_frames,
                      std::vector<float>& out_pcm);

    // Ultra-HD Fourier refinement: Unit-Circle Phase Normalization + +1.5 dB Studio Air Boost (7-11.2 kHz)
    static void refineSpectrumSIMD(float* mag, float* phase_x, float* phase_y, int num_bins);

    // Backwards-compatible wrapper to ProsodyController
    static void applyMoodAndSentenceProsody(std::vector<float>& f0, std::vector<float>& energy,
                                            ClauseType clause, const MoodParams& mood);

    static MoodParams getMoodParams(Mood mood) { return ProsodyController::getMoodParams(mood); }

    // Natural low-level formant-filtered breath sound for long pauses
    static std::vector<float> generateBreathNoise(int num_samples, float amplitude = 0.004f);

private:
    // Model & speaker loading helpers
    bool loadUniversalSpeaker();
    bool loadSubnet(const std::string& module_name,
                    std::unique_ptr<ncnn::Net>& net,
                    MmapBuffer& param_mmap,
                    MmapBuffer& bin_mmap);

    // Internal iSTFT tables & buffers
    void initISTFT();
    void radix2IFFT(float* real, float* imag, int n);

    EngineConfig m_config;
    SynthesisStats m_stats;
    bool m_is_initialized{false};

    // Step 1, 2 & 3: Linguistic normalizer, 21-lang colloquial slang & 62-D Articulatory mapper
    MultilingualG2PN m_g2pn;

    // Locked 192-D ECAPA-TDNN Vocal Cord Timbre Vector (`models/universal_speaker.bin`, 768 bytes)
    MmapBuffer m_mmap_speaker_bin;
    std::array<float, SPEAKER_EMBED_DIM> m_universal_speaker{};

    // 4 Modular NCNN Networks (IMS-Toucan + Vocos 24kHz Architecture)
    std::unique_ptr<ncnn::Net> m_net_toucan_encoder;     // Subnet 1:  toucan_encoder.bin (~9.5 MB FP16)
    std::unique_ptr<ncnn::Net> m_net_explicit_prosody;   // Subnet 2A: explicit_prosody.bin (~3.2 MB FP16)
    std::unique_ptr<ncnn::Net> m_net_toucan_cfm_decoder; // Subnet 2B: toucan_cfm_decoder.bin (~14.8 MB FP16)
    std::unique_ptr<ncnn::Net> m_net_vocos;              // Step 6:    vocos_hd_backbone.bin (27.47 MB)

    // Zero-copy OS memory-mapped file handles
    MmapBuffer m_mmap_encoder_param;
    MmapBuffer m_mmap_encoder_bin;
    MmapBuffer m_mmap_prosody_param;
    MmapBuffer m_mmap_prosody_bin;
    MmapBuffer m_mmap_cfm_param;
    MmapBuffer m_mmap_cfm_bin;
    MmapBuffer m_mmap_vocos_param;
    MmapBuffer m_mmap_vocos_bin;

    // Precomputed iSTFT tables
    std::vector<float> m_hann_window;
    std::vector<float> m_window_normalization;
    std::vector<float> m_twiddle_cos;
    std::vector<float> m_twiddle_sin;
    std::vector<int>   m_bit_reversal;

    // Step 8: Streaming state & 512-sample crossfade buffer
    std::vector<float> m_crossfade_tail;
};

} // namespace HDModularTTS
