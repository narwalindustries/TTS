#include "HDModularTTSEngine.h"
#include "miniaudio.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <cctype>

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #include <psapi.h>
    #pragma comment(lib, "psapi.lib")
#endif

namespace fs = std::filesystem;

struct LanguageSample {
    std::string code;
    std::string name;
    std::string default_mood;
    std::string text;
};

static const std::vector<LanguageSample> kAll21Languages = {
    {"en", "English",    "Casual_Slang", "Hey bro, what are you going to do today? Everything sounds super natural!"},
    {"hi", "Hindi",      "Casual_Slang", "अरे भाई! तू क्या कर रहा है? मुझे तो कुछ समझ नहीं आया, पर यह आवाज़ एकदम असली है!"},
    {"es", "Spanish",    "Excited",      "¡Hola amigos! ¿Qué tal están hoy? ¡La síntesis de voz suena increíblemente clara!"},
    {"fr", "French",     "Casual_Slang", "Franchement, je ne sais pas quoi dire, cette voix naturelle est vraiment incroyable !"},
    {"de", "German",     "Serious",      "Guten Tag, wir befinden uns jetzt in der hochauflösenden modularen Sprachsynthese."},
    {"zh", "Chinese",    "Happy",        "你好世界！欢迎使用高保真模块化语音合成系统，音质非常清晰自然！"},
    {"ja", "Japanese",   "Happy",        "こんにちは世界！高精細モジュラー音声合成システムへようこそ！"},
    {"ko", "Korean",     "Excited",      "안녕하세요 여러분! 고화질 텍스트 음성 변환 시스템이 정말 자연스럽게 들리네요!"},
    {"ru", "Russian",    "Storytelling", "Привет всем, добро пожаловать в модульную систему синтеза речи высокой четкости."},
    {"ar", "Arabic",     "Serious",      "مرحبا بالعالم، أهلا بكم في نظام تحويل النص إلى كلام عالي الدقة بصوت واضح وطبيعي."},
    {"pt", "Portuguese", "Happy",        "Olá a todos! Bem-vindos à síntese de voz modular em alta definição com clareza total!"},
    {"it", "Italian",    "Excited",      "Ciao a tutti! Benvenuti nella sintesi vocale modulare ad alta definizione!"},
    {"tr", "Turkish",    "Calm",         "Merhaba dünya, yüksek çözünürlüklü modüler metinden sese sistemine hoş geldiniz."},
    {"nl", "Dutch",      "Happy",        "Hallo allemaal, welkom bij hoogwaardige modulaire tekst naar spraak met natuurlijke intonatie!"},
    {"pl", "Polish",     "Storytelling", "Witaj świecie, witamy w modułowej syntezie mowy wysokiej rozdzielczości."},
    {"id", "Indonesian", "Casual_Slang", "Halo semuanya, selamat datang di sintesis suara modular definisi tinggi yang sangat jernih!"},
    {"vi", "Vietnamese", "Happy",        "Xin chào thế giới, chào mừng bạn đến với hệ thống chuyển văn bản thành giọng nói độ nét cao!"},
    {"th", "Thai",       "Calm",         "สวัสดีชาวโลก ยินดีต้อนรับสู่ระบบสังเคราะห์เสียงแบบแยกส่วนความคมชัดสูง"},
    {"ta", "Tamil",      "Casual_Slang", "வணக்கம் நண்பர்களே, நான் இப்போ தான் வந்தேன், இந்த குரல் ரொம்ப இயற்கையா இருக்கு!"},
    {"bn", "Bengali",    "Happy",        "হ্যালো বন্ধুরা, উচ্চ মানের মডুলার টেক্সট টু স্পিচ সিস্টেমে আপনাকে আন্তরিক স্বাগতম!"},
    {"ur", "Urdu",       "Casual_Slang", "ارے بھائی کیا کر رہا ہے؟ مجھے سمجھ نہیں آیا لیکن یہ آواز بالکل اصلی لگ رہی ہے!"}
};

HDModularTTS::Mood parseMoodString(const std::string& mood_str) {
    std::string lower = mood_str;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "happy") return HDModularTTS::Mood::Happy;
    if (lower == "serious") return HDModularTTS::Mood::Serious;
    if (lower == "storytelling") return HDModularTTS::Mood::Storytelling;
    if (lower == "calm") return HDModularTTS::Mood::Calm;
    if (lower == "excited") return HDModularTTS::Mood::Excited;
    if (lower == "casual_slang" || lower == "slang" || lower == "casual") return HDModularTTS::Mood::Casual_Slang;
    return HDModularTTS::Mood::Neutral;
}

std::string moodToString(HDModularTTS::Mood mood) {
    switch (mood) {
        case HDModularTTS::Mood::Happy: return "Happy";
        case HDModularTTS::Mood::Serious: return "Serious";
        case HDModularTTS::Mood::Storytelling: return "Storytelling";
        case HDModularTTS::Mood::Calm: return "Calm";
        case HDModularTTS::Mood::Excited: return "Excited";
        case HDModularTTS::Mood::Casual_Slang: return "Casual_Slang";
        default: return "Neutral";
    }
}

void printUsage(const char* prog) {
    std::cout << "========================================================================\n"
              << " IMS-Toucan 21-Language C++20 + Tencent NCNN CLI Speech Synthesizer\n"
              << " Single Locked Voice (universal_speaker.bin) | 62-D Articulatory + Moods\n"
              << "========================================================================\n"
              << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --text <string>        Text to speak instantly in any of the 21 languages\n"
              << "  --lang <code>          Language code (en,hi,es,fr,de,zh,ja,ko,ru,ar,pt,it,tr,nl,pl,id,vi,th,ta,bn,ur)\n"
              << "  --mood <name>          Mood/Emotion (Casual_Slang, Excited, Happy, Serious, Storytelling, Calm, Neutral)\n"
              << "  --all                  Synthesize all 21 languages with moods into ./output folder\n"
              << "  --interactive          Interactive terminal loop: type anything & hear it instantly\n"
              << "  --output_dir <path>    Directory for generated WAV outputs (default: './output')\n"
              << "  --out <wav_file>       Specific output WAV file path\n"
              << "  --models_dir <path>    Path to models directory (default: './models')\n"
              << "  --threads <n>          Number of CPU threads (default: 4)\n"
              << "  --vulkan               Enable Vulkan GPU acceleration\n"
              << "  --speed <float>        Prosody speech rate (default: 1.0)\n"
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
    std::string text;
    std::string lang = "hi";
    std::string mood_str = "Casual_Slang";
    std::string models_dir = "./models";
    std::string output_dir = "./output";
    std::string out_wav;
    int threads = 4;
    bool use_vulkan = false;
    float speed = 1.0f;
    bool batch_all = false;
    bool interactive = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--text" && i + 1 < argc) text = argv[++i];
        else if (arg == "--lang" && i + 1 < argc) lang = argv[++i];
        else if (arg == "--mood" && i + 1 < argc) mood_str = argv[++i];
        else if (arg == "--models_dir" && i + 1 < argc) models_dir = argv[++i];
        else if (arg == "--output_dir" && i + 1 < argc) output_dir = argv[++i];
        else if (arg == "--out" && i + 1 < argc) out_wav = argv[++i];
        else if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
        else if (arg == "--speed" && i + 1 < argc) speed = std::stof(argv[++i]);
        else if (arg == "--vulkan") use_vulkan = true;
        else if (arg == "--all") batch_all = true;
        else if (arg == "--interactive") interactive = true;
        else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
    }

    if (text.empty() && !interactive) {
        batch_all = true;
    }

    fs::create_directories(output_dir);

    HDModularTTS::EngineConfig config;
    config.models_dir = models_dir;
    config.speaker_bin_path = models_dir + "/universal_speaker.bin";
    config.num_threads = threads;
    config.use_vulkan = use_vulkan;
    config.sample_rate = 24000;
    config.speed_ratio = speed;
    config.crossfade_samples = 512;
    config.mood = parseMoodString(mood_str);

    HDModularTTS::HDModularTTSEngine engine(config);

    auto mmap_start = std::chrono::high_resolution_clock::now();
    if (!engine.init()) {
        std::cerr << "[FATAL] Failed to initialize HDModularTTSEngine from " << models_dir << std::endl;
        return 1;
    }
    auto mmap_end = std::chrono::high_resolution_clock::now();
    double init_time_ms = std::chrono::duration<double, std::milli>(mmap_end - mmap_start).count();

    std::cout << "========================================================================\n"
              << " IMS-Toucan 21-Language Single-Voice TTS Engine (C++20 + NCNN FP16)\n"
              << " Locked Voice:     " << config.speaker_bin_path << " (768 B / 192-D ECAPA-TDNN)\n"
              << " Startup (mmap):   " << std::fixed << std::setprecision(2) << init_time_ms << " ms\n"
              << " Output Folder:    " << output_dir << "\n"
              << "========================================================================\n";

    AudioDevice::RealtimeAudioPlayer audio_player(config.sample_rate, 1);

    auto speakAndSave = [&](const std::string& input_text,
                            const std::string& lang_code,
                            HDModularTTS::Mood mood,
                            const std::string& wav_path) -> bool {
        engine.setMood(mood);
        std::vector<float> full_audio;
        bool ok = engine.synthesizeStream(input_text, lang_code, [&](const std::vector<float>& pcm_chunk, bool /*is_final*/) {
            audio_player.playChunk(pcm_chunk);
            full_audio.insert(full_audio.end(), pcm_chunk.begin(), pcm_chunk.end());
        });
        if (!ok || full_audio.empty()) return false;
        AudioDevice::RealtimeAudioPlayer::writeWavFile(wav_path, full_audio, config.sample_rate);
        double dur_s = static_cast<double>(full_audio.size()) / config.sample_rate;
        std::cout << "  [OK] [" << lang_code << " | " << std::setw(12) << std::left << moodToString(mood) << "] -> "
                  << wav_path << " (" << std::fixed << std::setprecision(2) << dur_s << "s, "
                  << full_audio.size() << " samples @ 24kHz)\n";
        return true;
    };

    if (batch_all) {
        std::cout << "\n[*] Generating all 21 languages in a single locked voice into '" << output_dir << "'...\n";
        for (size_t idx = 0; idx < kAll21Languages.size(); ++idx) {
            const auto& item = kAll21Languages[idx];
            HDModularTTS::Mood m = parseMoodString(item.default_mood);
            std::ostringstream oss;
            oss << output_dir << "/" << std::setw(2) << std::setfill('0') << idx
                << "_" << item.code << "_" << item.default_mood << ".wav";
            speakAndSave(item.text, item.code, m, oss.str());
        }
    }

    if (!text.empty()) {
        std::string target_wav = out_wav.empty()
            ? (output_dir + "/" + lang + "_" + moodToString(config.mood) + "_instant.wav")
            : out_wav;
        speakAndSave(text, lang, config.mood, target_wav);
    }

    if (interactive) {
        std::cout << "\n[INTERACTIVE MODE] Type any sentence to speak instantly (or 'quit' to exit):\n";
        std::string line;
        int counter = 1;
        while (std::cout << "\n(" << lang << " | " << moodToString(config.mood) << ") > " && std::getline(std::cin, line)) {
            if (line == "quit" || line == "exit") break;
            if (line.empty()) continue;
            std::string wav_path = output_dir + "/interactive_" + std::to_string(counter++) + "_" + lang + ".wav";
            speakAndSave(line, lang, config.mood, wav_path);
        }
    }

    return 0;
}
