#pragma once

#include "MultilingualG2PN.h"
#include <vector>
#include <string>
#include <cstdint>

namespace HDModularTTS {

/**
 * Mood / Emotion Profiles for Native C++20 Expressiveness (Step 5B)
 * Includes both UPPER_CASE and Spec-Named aliases for seamless C++20 & JNI compatibility.
 */
enum class Mood {
    NEUTRAL = 0,
    HAPPY = 1,
    SERIOUS = 2,
    STORYTELLING = 3,
    CALM = 4,
    EXCITED = 5,
    CASUAL_SLANG = 6,

    // Spec-exact aliases
    Neutral = NEUTRAL,
    Happy = HAPPY,
    Serious = SERIOUS,
    Storytelling = STORYTELLING,
    Calm = CALM,
    Excited = EXCITED,
    Casual_Slang = CASUAL_SLANG
};

struct MoodParams {
    float pitch_shift{1.0f};     // Baseline F0 shift multiplier
    float variance_scale{1.0f};  // F0 variance multiplier around voiced mean
    float energy_scale{1.0f};    // Energy/loudness multiplier
    float duration_scale{1.0f};  // Duration multiplier (< 1.0 = faster cadence, > 1.0 = slower)
    float speed_scale{1.0f};     // Inverse duration speed multiplier
    std::string name{"neutral"};
};

/**
 * ProsodyController (Step 5B)
 * Native C++20 Mood, Sentence & Slang Pitch/Energy/Duration Controller.
 * Operates directly on the unlocked 1D tensors (`durations[N]`, `f0_pitch[N]`, `energy[N]`)
 * emitted by Subnet 2A (`explicit_prosody.bin`) before passing them to Subnet 2B (`toucan_cfm_decoder.bin`).
 */
class ProsodyController {
public:
    ProsodyController() = default;
    ~ProsodyController() = default;

    // Resolves Mood enum to acoustic scaling parameters
    static MoodParams getMoodParams(Mood mood) noexcept;

    // Checks if a specific token has lexical tone features active in Dims 55..61
    static bool hasLexicalToneFeature(const float* articulatory_vec62) noexcept;

    // Primary Step 5B Prosody Shaping Kernel:
    // Modifies durations[N], f0_pitch[N], and energy[N] in-place with Tonal Safety for zh/th/vi
    static void applyProsodyControl(std::vector<float>& durations,
                                    std::vector<float>& f0_pitch,
                                    std::vector<float>& energy,
                                    const std::vector<float>& articulatory_features,
                                    int32_t lang_id,
                                    ClauseType clause_type,
                                    Mood mood,
                                    float user_speed_ratio = 1.0f);
};

} // namespace HDModularTTS
