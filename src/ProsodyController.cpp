#include "ProsodyController.h"

#include <cmath>
#include <algorithm>

namespace HDModularTTS {

MoodParams ProsodyController::getMoodParams(Mood mood) noexcept {
    switch (mood) {
        case Mood::CASUAL_SLANG:
            // Casual_Slang: punchy colloquial cadence, expressive variance, slightly brisker duration
            return {1.03f, 1.22f, 1.06f, 0.92f, 1.09f, "casual_slang"};
        case Mood::EXCITED:
            // Excited: high pitch shift, wide variance, strong energy, fast pace
            return {1.12f, 1.45f, 1.18f, 0.88f, 1.14f, "excited"};
        case Mood::HAPPY:
            // Happy: bright pitch lift, lively variance, upbeat tempo
            return {1.08f, 1.30f, 1.08f, 0.94f, 1.06f, "happy"};
        case Mood::SERIOUS:
            // Serious: grounded lower F0, restrained variance, deliberate pacing
            return {0.94f, 0.72f, 0.96f, 1.06f, 0.94f, "serious"};
        case Mood::STORYTELLING:
            // Storytelling: rich dynamic contour, expressive pauses
            return {1.02f, 1.28f, 1.04f, 1.04f, 0.96f, "storytelling"};
        case Mood::CALM:
            // Calm: gentle lower F0, smooth contour, relaxed tempo
            return {0.96f, 0.78f, 0.90f, 1.08f, 0.92f, "calm"};
        case Mood::NEUTRAL:
        default:
            return {1.00f, 1.00f, 1.00f, 1.00f, 1.00f, "neutral"};
    }
}

bool ProsodyController::hasLexicalToneFeature(const float* articulatory_vec62) noexcept {
    if (!articulatory_vec62) return false;
    // Dims 55..61 encode lexical tones (T1..T5) and pitch accent steps
    for (int d = 55; d <= 61; ++d) {
        if (articulatory_vec62[d] > 0.01f) {
            return true;
        }
    }
    return false;
}

void ProsodyController::applyProsodyControl(std::vector<float>& durations,
                                            std::vector<float>& f0_pitch,
                                            std::vector<float>& energy,
                                            const std::vector<float>& articulatory_features,
                                            int32_t lang_id,
                                            ClauseType clause_type,
                                            Mood mood,
                                            float user_speed_ratio) {
    const size_t N = f0_pitch.size();
    if (N == 0) return;

    MoodParams mp = getMoodParams(mood);

    // Sentence-Type Exclamation ('!') macro overrides per Step 5B specification:
    // +12% pitch shift, 1.35x pitch variance, and 1.20x energy boost
    float effective_pitch_shift    = mp.pitch_shift;
    float effective_variance_scale = mp.variance_scale;
    float effective_energy_scale   = mp.energy_scale;

    if (clause_type == ClauseType::EXCLAMATION) {
        effective_pitch_shift    *= 1.12f;
        effective_variance_scale *= 1.35f;
        effective_energy_scale   *= 1.20f;
    }

    // Check Tonal Safety requirement:
    // zh (5), vi (16), th (17) are lexical tone languages where local token-to-token
    // tone contours (Dims 55..61) determine word meaning.
    const bool is_tonal_lang = (lang_id == 5 || lang_id == 16 || lang_id == 17);

    // 1. Compute voiced F0 mean across the clause
    float voiced_sum = 0.0f;
    std::vector<size_t> voiced_indices;
    voiced_indices.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        if (f0_pitch[i] > 50.0f) {
            voiced_sum += f0_pitch[i];
            voiced_indices.push_back(i);
        }
    }

    const float mean_f0 = voiced_indices.empty() ? 210.0f : (voiced_sum / static_cast<float>(voiced_indices.size()));

    // 2. Scale durations [N] according to Mood duration_scale and user_speed_ratio
    const float safe_speed = std::max(0.25f, std::min(4.0f, user_speed_ratio));
    const float dur_multiplier = mp.duration_scale / safe_speed;
    for (size_t i = 0; i < durations.size(); ++i) {
        durations[i] = std::max(1.0f, durations[i] * dur_multiplier);
    }

    // 3. Apply Mood F0 shift, variance around voiced mean, and Tonal-Safe Sentence Intonation
    const size_t num_voiced = voiced_indices.size();
    for (size_t v_rank = 0; v_rank < num_voiced; ++v_rank) {
        const size_t idx = voiced_indices[v_rank];
        const float raw_f0 = f0_pitch[idx];

        const float* art_vec = (articulatory_features.size() >= (idx + 1) * MultilingualG2PN::ARTICULATORY_DIM)
            ? &articulatory_features[idx * MultilingualG2PN::ARTICULATORY_DIM]
            : nullptr;
        const bool token_has_tone = hasLexicalToneFeature(art_vec);

        float shaped_f0 = raw_f0;

        if (is_tonal_lang || token_has_tone) {
            // TONAL SAFETY (zh, th, vi):
            // Preserve local token-to-token lexical tone contours (Dims 55-61) intact
            // while shifting only the global phrase register!
            float local_tone_delta = raw_f0 - mean_f0;

            // Reinforce explicit lexical tone targets from Dims 55..59 if present
            if (art_vec != nullptr) {
                if (art_vec[55] > 0.5f) {
                    // Tone 1 (High level ˥)
                    local_tone_delta = std::max(local_tone_delta, 0.14f * mean_f0);
                } else if (art_vec[56] > 0.5f) {
                    // Tone 2 (High rising ˦)
                    local_tone_delta += 0.08f * mean_f0;
                } else if (art_vec[57] > 0.5f) {
                    // Tone 3 (Dipping low ˧)
                    local_tone_delta = std::min(local_tone_delta, -0.10f * mean_f0);
                } else if (art_vec[58] > 0.5f) {
                    // Tone 4 (High falling ˨)
                    local_tone_delta -= 0.06f * mean_f0;
                }
            }

            // Shift global phrase register by effective_pitch_shift without distorting local_tone_delta
            float global_register_shift = mean_f0 * effective_pitch_shift;
            if (clause_type == ClauseType::QUESTION) {
                global_register_shift *= 1.06f; // Gentle global register lift for tonal questions
            } else if (clause_type == ClauseType::STATEMENT) {
                global_register_shift *= 0.98f;
            }
            shaped_f0 = global_register_shift + local_tone_delta;
        } else {
            // NON-TONAL LANGUAGES:
            // Scale baseline F0 shift and F0 variance around voiced mean
            shaped_f0 = (mean_f0 * effective_pitch_shift) +
                        effective_variance_scale * (raw_f0 - mean_f0);

            // Normalized position within voiced tokens [0.0 .. 1.0]
            const float voiced_pos = (num_voiced > 1)
                ? (static_cast<float>(v_rank) / static_cast<float>(num_voiced - 1))
                : 1.0f;

            if (clause_type == ClauseType::QUESTION) {
                // '?' (Question): Smooth +22% quadratic pitch rise on the last 20% voiced tokens
                if (voiced_pos >= 0.80f) {
                    const float t = (voiced_pos - 0.80f) / 0.20f;
                    shaped_f0 *= (1.0f + 0.22f * (t * t));
                }
            } else if (clause_type == ClauseType::COMMA) {
                // ',' (Continuation): Slight +6% pre-boundary continuation rise before 180ms pause
                if (voiced_pos >= 0.75f) {
                    const float t = (voiced_pos - 0.75f) / 0.25f;
                    shaped_f0 *= (1.0f + 0.06f * t);
                }
            } else if (clause_type == ClauseType::STATEMENT) {
                // '.' (Statement): Natural -8% pitch declination at clause end before 320ms pause
                if (voiced_pos >= 0.80f) {
                    const float t = (voiced_pos - 0.80f) / 0.20f;
                    shaped_f0 *= (1.0f - 0.08f * t);
                }
            }
        }

        f0_pitch[idx] = std::max(55.0f, std::min(650.0f, shaped_f0));
    }

    // 4. Scale energy [N] according to Mood & Sentence-Type Exclamation boost
    for (size_t i = 0; i < energy.size(); ++i) {
        energy[i] = std::max(0.01f, energy[i] * effective_energy_scale);
    }
}

} // namespace HDModularTTS
