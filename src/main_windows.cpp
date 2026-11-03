#include "HDModularTTSEngine.h"
#include "miniaudio.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #include <psapi.h>
    #pragma comment(lib, "psapi.lib")
#endif

void printUsage(const char* prog) {
    std::cout << "========================================================================\n"
              << " HD Modular Multilingual TTS - Tencent NCNN Windows Architecture\n"
              << " Decoupled Prosody & Locked Speaker Timbre | Vocos HD Fourier Head\n"
              << "========================================================================\n"
              << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --text <string>        Text to synthesize (default: 'Hello world, welcome to high-definition modular TTS.')\n"
              << "  --lang <code>          Language ISO code (en, hi, es, fr, de, zh, ja, etc., default: 'en')\n"
              << "  --models_dir <path>    Path to models directory (default: './models')\n"
              << "  --threads <n>          Number of CPU threads (default: 4)\n"
              << "  --vulkan               Enable Vulkan GPU acceleration (default: off)\n"
              << "  --speed <float>        Prosody speech rate (default: 1.0)\n"
              << "  --out <wav_file>       Save synthesized audio to WAV file\n"
              << "  --benchmark            Print detailed latency and memory telemetry\n"
              << "  --help                 Show this help message\n"
              << "========================================================================\n";
}

size_t getProcessWorkingSetKB() {
#if defined(_WIN32) || defined(_WIN64)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / 1024;
    }
#endif
    return 0;
}

int main(int argc, char** argv) {
    std::string text = "Hello world, welcome to high-definition modular text-to-speech with crystal clear audio.";
    std::string lang = "en";
    std::string models_dir = "./models";
    std::string out_wav = "output.wav";
    int threads = 4;
    bool use_vulkan = false;
    float speed = 1.0f;
    bool benchmark = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--text" && i + 1 < argc) text = argv[++i];
        else if (arg == "--lang" && i + 1 < argc) lang = argv[++i];
        else if (arg == "--models_dir" && i + 1 < argc) models_dir = argv[++i];
        else if (arg == "--out" && i + 1 < argc) out_wav = argv[++i];
        else if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
        else if (arg == "--speed" && i + 1 < argc) speed = std::stof(argv[++i]);
        else if (arg == "--vulkan") use_vulkan = true;
        else if (arg == "--benchmark") benchmark = true;
        else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
    }

    std::cout << "\n=======================================================\n"
              << " HD Modular TTS Engine (Windows x64 Native C++20)\n"
              << " Target Language:  [" << lang << "]\n"
              << " Input Text:       \"" << text << "\"\n"
              << " Vulkan Compute:   " << (use_vulkan ? "ENABLED" : "DISABLED") << "\n"
              << " CPU Threads:      " << threads << "\n"
              << " Initial RAM:      " << getProcessWorkingSetKB() << " KB\n"
              << "=======================================================\n" << std::endl;

    HDModularTTS::EngineConfig config;
    config.models_dir = models_dir;
    config.num_threads = threads;
    config.use_vulkan = use_vulkan;
    config.sample_rate = 24000;
    config.speed_ratio = speed;
    config.crossfade_samples = 512;

    HDModularTTS::HDModularTTSEngine engine(config);

    auto mmap_start = std::chrono::high_resolution_clock::now();
    if (!engine.init()) {
        std::cerr << "[FATAL] Failed to initialize HDModularTTSEngine from " << models_dir << std::endl;
        return 1;
    }
    auto mmap_end = std::chrono::high_resolution_clock::now();
    double init_time_ms = std::chrono::duration<double, std::milli>(mmap_end - mmap_start).count();

    std::cout << "[INFO] Zero-Copy Model Startup Time (mmap): " << std::fixed << std::setprecision(2)
              << init_time_ms << " ms" << std::endl;
    std::cout << "[INFO] Post-mmap RAM: " << getProcessWorkingSetKB() << " KB (<150 MB target)\n" << std::endl;

    AudioDevice::RealtimeAudioPlayer audio_player(config.sample_rate, 1);
    std::vector<float> full_audio;

    std::cout << "[STREAMING] Starting real-time clause synthesis..." << std::endl;

    int chunk_index = 0;
    bool success = engine.synthesizeStream(text, lang, [&](const std::vector<float>& pcm_chunk, bool is_final) {
        chunk_index++;
        std::cout << "  -> Synthesized Chunk #" << chunk_index 
                  << " (" << pcm_chunk.size() << " samples, "
                  << static_cast<float>(pcm_chunk.size()) / config.sample_rate * 1000.0f << " ms)" 
                  << (is_final ? " [FINAL]" : "") << std::endl;

        // Immediate real-time playback
        audio_player.playChunk(pcm_chunk);

        // Accumulate for saving to WAV
        full_audio.insert(full_audio.end(), pcm_chunk.begin(), pcm_chunk.end());
    });

    if (!success) {
        std::cerr << "[ERROR] Synthesis failed!" << std::endl;
        return 1;
    }

    // Save to WAV file
    if (!out_wav.empty() && !full_audio.empty()) {
        if (AudioDevice::RealtimeAudioPlayer::writeWavFile(out_wav, full_audio, config.sample_rate)) {
            std::cout << "\n[INFO] Wrote synthesized HD audio to: " << out_wav << " (" << full_audio.size() << " samples)" << std::endl;
        }
    }

    if (benchmark) {
        const auto& stats = engine.getStats();
        std::cout << "\n=======================================================\n"
                  << " LATENCY & PERFORMANCE TELEMETRY BENCHMARK\n"
                  << "=======================================================\n"
                  << " Time-To-First-Audio (TTFA):   " << stats.time_to_first_audio_ms << " ms "
                  << (stats.time_to_first_audio_ms < 80.0 ? "[PASS: < 80ms CPU]" : "[WARN: > 80ms]") << "\n"
                  << " Total Synthesis Duration:     " << stats.total_synthesis_time_ms << " ms\n"
                  << " Generated Audio Length:       " << stats.total_audio_duration_sec << " s\n"
                  << " Real-Time Factor (RTF):       " << stats.real_time_factor << "x (Lower is faster)\n"
                  << " Total Clauses Processed:      " << stats.total_clauses_processed << "\n"
                  << " Total Samples Synthesized:    " << stats.total_samples_synthesized << "\n"
                  << " Peak Process Memory:          " << getProcessWorkingSetKB() << " KB\n"
                  << "=======================================================\n" << std::endl;
    }

    // Wait briefly for audio playback to complete before exit
    if (!full_audio.empty()) {
        int sleep_ms = static_cast<int>((static_cast<double>(full_audio.size()) / config.sample_rate) * 1000.0);
        if (sleep_ms > 0 && sleep_ms < 10000) {
            std::cout << "[INFO] Finishing playback..." << std::endl;
#if defined(_WIN32) || defined(_WIN64)
            Sleep(std::min(sleep_ms, 2500));
#endif
        }
    }

    return 0;
}

