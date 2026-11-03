#pragma once

#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <cstdint>
#include <string_view>

namespace HDModularTTS {

/**
 * Clause intonation and sentence classification (Step 3 Metadata)
 */
enum class ClauseType {
    STATEMENT,   // Period '.' or Hindi Danda '।' (-8% declination, 320ms pause)
    QUESTION,    // Question '?' (+22% quadratic rising intonation on last 20% voiced tokens)
    EXCLAMATION, // Exclamation '!' (+12% pitch shift, 1.35x variance, 1.20x energy)
    COMMA        // Comma ',' or semicolon ';' (+6% continuation rise, 180ms pause)
};

/**
 * ClauseSegment with intonation tags and breathing duration
 */
struct ClauseSegment {
    std::string text;
    ClauseType type{ClauseType::STATEMENT};
    float pause_sec{0.18f};
    bool has_breath{false};
};

/**
 * PhonemeSequence represents processed linguistic tokens and 62-D Articulatory
 * Feature Vectors ready for Subnet 1 (toucan_encoder.bin).
 */
struct PhonemeSequence {
    std::vector<int32_t> phoneme_ids;
    int32_t lang_id{0};
    int32_t accent_style_id{0};
    ClauseType clause_type{ClauseType::STATEMENT};
    float pause_sec{0.32f};
    bool is_tonal_language{false};
    std::string text_clause;
    std::vector<float> articulatory_features; // Flattened [N, 62] float matrix
};

/**
 * Compile-time 62-D Articulatory Feature Archetype Table (~15.5 KB in .rodata)
 * 64 canonical PanPhon/IMS-Toucan phoneme archetypes x 62 floats x 4 bytes = 15,872 bytes.
 *
 * Dimension Layout (float[62]):
 *   Dims 0..21:  Vocal Tract Place of Articulation:
 *                [0] syllabic, [1] consonantal, [2] sonorant, [3] continuant,
 *                [4] delayed_release, [5] lateral, [6] nasal, [7] labial,
 *                [8] bilabial, [9] labiodental, [10] coronal, [11] dental (HARD-LOCKED त/थ/द/ध),
 *                [12] alveolar (English t/d), [13] postalveolar, [14] retroflex (HARD-LOCKED ट/ठ/ड/ढ/ழ),
 *                [15] dorsal, [16] palatal, [17] velar, [18] uvular (HARD-LOCKED क़/ख़/ग़/ʁ),
 *                [19] pharyngeal (HARD-LOCKED Arabic ع/ح & Emphatics), [20] glottal, [21] epilaryngeal
 *   Dims 22..35: Manner & Stricture Modifiers:
 *                [22] stop/plosive, [23] fricative, [24] affricate, [25] approximant,
 *                [26] trill, [27] tap/flap (English Flapped-T ɾ / Indic ɽ), [28] sibilant,
 *                [29] strident, [30] geminate/long_consonant, [31] synalepha_linked,
 *                [32] syllable_timed_rhythm, [33] stress_timed_rhythm, [34] mora_timed_rhythm,
 *                [35] schwa_deleted_boundary
 *   Dims 36..40: Laryngeal, Voicing & Articulatory Tension:
 *                [36] voiced, [37] spread_glottis/aspirated, [38] constricted_glottis/ejective,
 *                [39] fortis_tense (Korean ㅃ/ㄸ/ㄲ/ㅆ/ㅉ), [40] lenis_lax / breathy_voiced (भ/ध/घ/ढ)
 *   Dims 41..54: Vowel Space & Secondary Articulation:
 *                [41] high/close, [42] mid, [43] low/open, [44] front, [45] central,
 *                [46] back, [47] rounded (Bengali অ=/ɔ/), [48] tense_vowel, [49] nasalized_vowel (Tamil/Hindi/French ̃),
 *                [50] long_vowel (ː), [51] labialized (ʷ), [52] palatalized (ʲ),
 *                [53] velarized/emphatic (ˤ - Arabic ص/ض/ط/ظ), [54] rhotacized
 *   Dims 55..61: Lexical Tone & Pitch Accent Contours (Preserved by ProsodyController):
 *                [55] tone_high_level (T1 / ˥), [56] tone_rising (T2 / ˦),
 *                [57] tone_dipping_low (T3 / ˧), [58] tone_falling (T4 / ˨),
 *                [59] tone_neutral_low (T5 / ˩), [60] pitch_accent_upstep (Japanese ꜛ / Vietnamese broken),
 *                [61] pitch_accent_downstep (Japanese ꜜ / glottalized tone)
 */
namespace ArticulatoryLUT {

    static constexpr size_t NUM_ARCHETYPES = 64;
    static constexpr size_t DIM = 62;

    using FeatureVec = std::array<float, DIM>;
    using TableType  = std::array<FeatureVec, NUM_ARCHETYPES>;

    consteval TableType buildConstexprTable() {
        TableType t{};
        for (size_t i = 0; i < NUM_ARCHETYPES; ++i) {
            for (size_t d = 0; d < DIM; ++d) {
                t[i][d] = 0.0f;
            }
        }

        // 0: PAD / Silence
        // 1: BOS, 2: EOS, 3: PAUSE_SHORT (100ms), 4: PAUSE_MEDIUM (180ms), 5: PAUSE_LONG (320ms)

        // --- VOWELS (Archetypes 6..19) ---
        // 6: 'a' (Open front/central unrounded)
        t[6][0] = 1.0f; t[6][2] = 1.0f; t[6][3] = 1.0f; t[6][36] = 1.0f; t[6][43] = 1.0f; t[6][44] = 1.0f;
        // 7: 'ɑ' / 'ɑː' (Open back unrounded)
        t[7][0] = 1.0f; t[7][2] = 1.0f; t[7][3] = 1.0f; t[7][36] = 1.0f; t[7][43] = 1.0f; t[7][46] = 1.0f; t[7][50] = 1.0f;
        // 8: 'æ' (Near-open front unrounded)
        t[8][0] = 1.0f; t[8][2] = 1.0f; t[8][3] = 1.0f; t[8][36] = 1.0f; t[8][43] = 1.0f; t[8][44] = 1.0f; t[8][48] = 0.5f;
        // 9: 'ə' (Mid central schwa)
        t[9][0] = 1.0f; t[9][2] = 1.0f; t[9][3] = 1.0f; t[9][36] = 1.0f; t[9][42] = 1.0f; t[9][45] = 1.0f;
        // 10: 'e' / 'eː' (Close-mid front unrounded)
        t[10][0] = 1.0f; t[10][2] = 1.0f; t[10][3] = 1.0f; t[10][36] = 1.0f; t[10][42] = 1.0f; t[10][44] = 1.0f; t[10][48] = 1.0f;
        // 11: 'ɛ' (Open-mid front unrounded)
        t[11][0] = 1.0f; t[11][2] = 1.0f; t[11][3] = 1.0f; t[11][36] = 1.0f; t[11][42] = 1.0f; t[11][44] = 1.0f;
        // 12: 'i' / 'iː' (Close front unrounded)
        t[12][0] = 1.0f; t[12][2] = 1.0f; t[12][3] = 1.0f; t[12][36] = 1.0f; t[12][41] = 1.0f; t[12][44] = 1.0f; t[12][48] = 1.0f;
        // 13: 'ɪ' (Near-close front unrounded)
        t[13][0] = 1.0f; t[13][2] = 1.0f; t[13][3] = 1.0f; t[13][36] = 1.0f; t[13][41] = 1.0f; t[13][44] = 1.0f;
        // 14: 'o' / 'oː' (Close-mid back rounded)
        t[14][0] = 1.0f; t[14][2] = 1.0f; t[14][3] = 1.0f; t[14][36] = 1.0f; t[14][42] = 1.0f; t[14][46] = 1.0f; t[14][47] = 1.0f; t[14][48] = 1.0f;
        // 15: 'ɔ' (Open-mid back rounded - Bengali inherent vowel)
        t[15][0] = 1.0f; t[15][2] = 1.0f; t[15][3] = 1.0f; t[15][36] = 1.0f; t[15][42] = 1.0f; t[15][46] = 1.0f; t[15][47] = 1.0f;
        // 16: 'u' / 'uː' (Close back rounded)
        t[16][0] = 1.0f; t[16][2] = 1.0f; t[16][3] = 1.0f; t[16][36] = 1.0f; t[16][41] = 1.0f; t[16][46] = 1.0f; t[16][47] = 1.0f; t[16][48] = 1.0f;
        // 17: 'ʊ' (Near-close back rounded)
        t[17][0] = 1.0f; t[17][2] = 1.0f; t[17][3] = 1.0f; t[17][36] = 1.0f; t[17][41] = 1.0f; t[17][46] = 1.0f; t[17][47] = 1.0f;
        // 18: 'y' / 'ø' / 'œ' (Front rounded vowels - French/German/Turkish)
        t[18][0] = 1.0f; t[18][2] = 1.0f; t[18][3] = 1.0f; t[18][36] = 1.0f; t[18][41] = 1.0f; t[18][44] = 1.0f; t[18][47] = 1.0f;
        // 19: 'ɨ' / 'ɯ' / 'ʌ' / 'ɤ' (Close/mid back/central unrounded - Russian/Turkish/Korean/Thai/Vietnamese)
        t[19][0] = 1.0f; t[19][2] = 1.0f; t[19][3] = 1.0f; t[19][36] = 1.0f; t[19][41] = 1.0f; t[19][45] = 1.0f;

        // --- LABIAL CONSONANTS (Archetypes 20..25) ---
        // 20: 'p' (Voiceless bilabial stop)
        t[20][1] = 1.0f; t[20][7] = 1.0f; t[20][8] = 1.0f; t[20][22] = 1.0f;
        // 21: 'pʰ' (Aspirated voiceless bilabial stop - ফ/फ)
        t[21][1] = 1.0f; t[21][7] = 1.0f; t[21][8] = 1.0f; t[21][22] = 1.0f; t[21][37] = 1.0f;
        // 22: 'b' (Voiced bilabial stop)
        t[22][1] = 1.0f; t[22][7] = 1.0f; t[22][8] = 1.0f; t[22][22] = 1.0f; t[22][36] = 1.0f;
        // 23: 'bʱ' (Breathy-voiced bilabial stop - भ)
        t[23][1] = 1.0f; t[23][7] = 1.0f; t[23][8] = 1.0f; t[23][22] = 1.0f; t[23][36] = 1.0f; t[23][37] = 1.0f; t[23][40] = 1.0f;
        // 24: 'm' (Bilabial nasal)
        t[24][1] = 1.0f; t[24][2] = 1.0f; t[24][6] = 1.0f; t[24][7] = 1.0f; t[24][8] = 1.0f; t[24][36] = 1.0f;
        // 25: 'f' / 'v' (Labiodental fricative - Urdu फ़ locked)
        t[25][1] = 1.0f; t[25][3] = 1.0f; t[25][7] = 1.0f; t[25][9] = 1.0f; t[25][23] = 1.0f;

        // --- HARD-LOCKED DENTAL CONSONANTS (Archetypes 26..30: Hindi त, थ, द, ध / Tamil த / θ, ð) ---
        // Zero American Alveolar [12] bleed: Dim[11]=1.0 (Dental), Dim[12]=0.0 (Alveolar)!
        // 26: 't̪' (Voiceless dental plosive - त / த)
        t[26][1] = 1.0f; t[26][10] = 1.0f; t[26][11] = 1.0f; t[26][22] = 1.0f;
        // 27: 't̪ʰ' (Aspirated voiceless dental plosive - थ)
        t[27][1] = 1.0f; t[27][10] = 1.0f; t[27][11] = 1.0f; t[27][22] = 1.0f; t[27][37] = 1.0f;
        // 28: 'd̪' (Voiced dental plosive - द)
        t[28][1] = 1.0f; t[28][10] = 1.0f; t[28][11] = 1.0f; t[28][22] = 1.0f; t[28][36] = 1.0f;
        // 29: 'd̪ʱ' (Breathy-voiced dental plosive - ध)
        t[29][1] = 1.0f; t[29][10] = 1.0f; t[29][11] = 1.0f; t[29][22] = 1.0f; t[29][36] = 1.0f; t[29][37] = 1.0f; t[29][40] = 1.0f;
        // 30: 'θ' / 'ð' (Dental fricative)
        t[30][1] = 1.0f; t[30][3] = 1.0f; t[30][10] = 1.0f; t[30][11] = 1.0f; t[30][23] = 1.0f;

        // --- ALVEOLAR CONSONANTS (Archetypes 31..37: English t, d, Flapped-T ɾ, s, z, n, l, r) ---
        // 31: 't' (Voiceless alveolar plosive)
        t[31][1] = 1.0f; t[31][10] = 1.0f; t[31][12] = 1.0f; t[31][22] = 1.0f;
        // 32: 'd' (Voiced alveolar plosive)
        t[32][1] = 1.0f; t[32][10] = 1.0f; t[32][12] = 1.0f; t[32][22] = 1.0f; t[32][36] = 1.0f;
        // 33: 'ɾ' (Alveolar tap/flap - American English intervocalic Flapped-T / Spanish single-r)
        t[33][1] = 1.0f; t[33][2] = 1.0f; t[33][3] = 1.0f; t[33][10] = 1.0f; t[33][12] = 1.0f; t[33][27] = 1.0f; t[33][36] = 1.0f;
        // 34: 's' / 'z' (Alveolar sibilant fricative - Urdu ज़ locked)
        t[34][1] = 1.0f; t[34][3] = 1.0f; t[34][10] = 1.0f; t[34][12] = 1.0f; t[34][23] = 1.0f; t[34][28] = 1.0f; t[34][29] = 1.0f;
        // 35: 'n' (Alveolar nasal)
        t[35][1] = 1.0f; t[35][2] = 1.0f; t[35][6] = 1.0f; t[35][10] = 1.0f; t[35][12] = 1.0f; t[35][36] = 1.0f;
        // 36: 'l' (Alveolar lateral approximant)
        t[36][1] = 1.0f; t[36][2] = 1.0f; t[36][3] = 1.0f; t[36][5] = 1.0f; t[36][10] = 1.0f; t[36][12] = 1.0f; t[36][25] = 1.0f; t[36][36] = 1.0f;
        // 37: 'r' (Alveolar trill / approximant)
        t[37][1] = 1.0f; t[37][2] = 1.0f; t[37][3] = 1.0f; t[37][10] = 1.0f; t[37][12] = 1.0f; t[37][26] = 1.0f; t[37][36] = 1.0f;

        // --- HARD-LOCKED RETROFLEX CONSONANTS (Archetypes 38..45: Hindi ट, ठ, ड, ढ, ण, ड़ / Tamil ழ, ட, ண, ள) ---
        // Dim[14]=1.0 (Retroflex), Dim[12]=0.0 (Alveolar), Dim[11]=0.0 (Dental)
        // 38: 'ʈ' (Voiceless retroflex plosive - ट / ட)
        t[38][1] = 1.0f; t[38][10] = 1.0f; t[38][14] = 1.0f; t[38][22] = 1.0f;
        // 39: 'ʈʰ' (Aspirated voiceless retroflex plosive - ठ)
        t[39][1] = 1.0f; t[39][10] = 1.0f; t[39][14] = 1.0f; t[39][22] = 1.0f; t[39][37] = 1.0f;
        // 40: 'ɖ' (Voiced retroflex plosive - ड)
        t[40][1] = 1.0f; t[40][10] = 1.0f; t[40][14] = 1.0f; t[40][22] = 1.0f; t[40][36] = 1.0f;
        // 41: 'ɖʱ' (Breathy-voiced retroflex plosive - ढ)
        t[41][1] = 1.0f; t[41][10] = 1.0f; t[41][14] = 1.0f; t[41][22] = 1.0f; t[41][36] = 1.0f; t[41][37] = 1.0f; t[41][40] = 1.0f;
        // 42: 'ɳ' (Retroflex nasal - ण / ண)
        t[42][1] = 1.0f; t[42][2] = 1.0f; t[42][6] = 1.0f; t[42][10] = 1.0f; t[42][14] = 1.0f; t[42][36] = 1.0f;
        // 43: 'ɽ' (Retroflex flap - ड़ / ढ़)
        t[43][1] = 1.0f; t[43][2] = 1.0f; t[43][10] = 1.0f; t[43][14] = 1.0f; t[43][27] = 1.0f; t[43][36] = 1.0f;
        // 44: 'ɭ' (Retroflex lateral approximant - ळ / ள)
        t[44][1] = 1.0f; t[44][2] = 1.0f; t[44][3] = 1.0f; t[44][5] = 1.0f; t[44][10] = 1.0f; t[44][14] = 1.0f; t[44][25] = 1.0f; t[44][36] = 1.0f;
        // 45: 'ɻ' (HARD-LOCKED Tamil Retroflex Approximant ழ / Mandarin r)
        t[45][1] = 1.0f; t[45][2] = 1.0f; t[45][3] = 1.0f; t[45][5] = 1.0f; t[45][10] = 1.0f; t[45][14] = 1.0f; t[45][25] = 1.0f; t[45][36] = 1.0f; t[45][54] = 1.0f;

        // --- POSTALVEOLAR, PALATAL & VELAR CONSONANTS (Archetypes 46..53) ---
        // 46: 'ʃ' / 'ʒ' / 'ʂ' / 'ʐ' (Postalveolar/retroflex sibilant)
        t[46][1] = 1.0f; t[46][3] = 1.0f; t[46][10] = 1.0f; t[46][13] = 1.0f; t[46][23] = 1.0f; t[46][28] = 1.0f; t[46][29] = 1.0f;
        // 47: 'tʃ' / 'dʒ' / 'tʃʰ' / 'dʒʱ' (Postalveolar affricates - च, छ, ज, झ)
        t[47][1] = 1.0f; t[47][4] = 1.0f; t[47][10] = 1.0f; t[47][13] = 1.0f; t[47][24] = 1.0f; t[47][28] = 1.0f;
        // 48: 'j' / 'w' (Palatal / Labial-velar glides)
        t[48][2] = 1.0f; t[48][3] = 1.0f; t[48][15] = 1.0f; t[48][16] = 1.0f; t[48][25] = 1.0f; t[48][36] = 1.0f;
        // 49: 'k' / 'kʰ' (Voiceless velar plosive - क, ख)
        t[49][1] = 1.0f; t[49][15] = 1.0f; t[49][17] = 1.0f; t[49][22] = 1.0f;
        // 50: 'g' / 'gʱ' (Voiced velar plosive - ग, घ)
        t[50][1] = 1.0f; t[50][15] = 1.0f; t[50][17] = 1.0f; t[50][22] = 1.0f; t[50][36] = 1.0f;
        // 51: 'ŋ' / 'ɲ' (Velar / Palatal nasal)
        t[51][1] = 1.0f; t[51][2] = 1.0f; t[51][6] = 1.0f; t[51][15] = 1.0f; t[51][17] = 1.0f; t[51][36] = 1.0f;
        // 52: 'x' / 'ɣ' (HARD-LOCKED Urdu/Arabic velar-uvular fricatives ख़ / ग़)
        t[52][1] = 1.0f; t[52][3] = 1.0f; t[52][15] = 1.0f; t[52][17] = 1.0f; t[52][18] = 1.0f; t[52][23] = 1.0f;
        // 53: 'h' / 'ʔ' (Glottal fricative / stop)
        t[53][1] = 1.0f; t[53][3] = 1.0f; t[53][20] = 1.0f; t[53][23] = 1.0f;

        // --- HARD-LOCKED UVULAR, PHARYNGEAL & EMPHATIC CONSONANTS (Archetypes 54..58) ---
        // 54: 'q' (HARD-LOCKED Voiceless uvular plosive - Urdu/Arabic क़ / ق)
        t[54][1] = 1.0f; t[54][15] = 1.0f; t[54][18] = 1.0f; t[54][22] = 1.0f;
        // 55: 'ʁ' (HARD-LOCKED French Uvular-R / Arabic غ)
        t[55][1] = 1.0f; t[55][2] = 1.0f; t[55][3] = 1.0f; t[55][15] = 1.0f; t[55][18] = 1.0f; t[55][23] = 1.0f; t[55][36] = 1.0f;
        // 56: 'ħ' (HARD-LOCKED Voiceless pharyngeal fricative - Arabic ح)
        t[56][1] = 1.0f; t[56][3] = 1.0f; t[56][19] = 1.0f; t[56][21] = 1.0f; t[56][23] = 1.0f;
        // 57: 'ʕ' (HARD-LOCKED Voiced pharyngeal fricative - Arabic ع)
        t[57][1] = 1.0f; t[57][3] = 1.0f; t[57][19] = 1.0f; t[57][21] = 1.0f; t[57][23] = 1.0f; t[57][36] = 1.0f;
        // 58: 'sˤ' / 'dˤ' / 'tˤ' / 'ðˤ' (HARD-LOCKED Arabic Emphatic Pharyngealized Coronal ص, ض, ط, ظ)
        t[58][1] = 1.0f; t[58][10] = 1.0f; t[58][12] = 1.0f; t[58][19] = 1.0f; t[58][53] = 1.0f;

        // --- LEXICAL TONE & PROSODIC ARCHETYPES (Archetypes 59..63: Dims 55..61) ---
        // 59: Tone 1 (High Level ˥ - Chinese T1, Thai Mid/High, Vietnamese Ngang)
        t[59][55] = 1.0f;
        // 60: Tone 2 (High Rising ˦ - Chinese T2, Thai Rising, Vietnamese Sắc)
        t[60][56] = 1.0f;
        // 61: Tone 3 (Dipping / Low ˧ - Chinese T3, Thai Low, Vietnamese Hỏi/Ngã)
        t[61][57] = 1.0f;
        // 62: Tone 4 (High Falling ˨ - Chinese T4, Thai Falling, Vietnamese Huyền/Nặng)
        t[62][58] = 1.0f;
        // 63: Tone 5 / Pitch Accent Step (Neutral ˩ / Japanese High-Low Step ꜛ/ꜜ)
        t[63][59] = 1.0f; t[63][60] = 1.0f;

        return t;
    }

    // 64 * 62 * 4 = 15,872 bytes (~15.5 KB) compile-time constexpr table
    inline constexpr TableType kConstexprArticulatoryTable = buildConstexprTable();
    static_assert(sizeof(kConstexprArticulatoryTable) == 15872,
                  "Articulatory constexpr lookup table must be exactly 15,872 bytes (~15.5 KB)");

} // namespace ArticulatoryLUT

/**
 * MultilingualG2PN
 * Production C++20 Multilingual Text Normalizer, 21-Language Colloquial Slang
 * Reducer & 62-D Articulatory Feature Mapper (IMS-Toucan Architecture).
 * Supports all 21 Languages:
 *   0: en (English),   1: hi (Hindi),      2: es (Spanish),    3: fr (French),
 *   4: de (German),    5: zh (Chinese),    6: ja (Japanese),   7: ko (Korean),
 *   8: ru (Russian),   9: ar (Arabic),    10: pt (Portuguese), 11: it (Italian),
 *  12: tr (Turkish),  13: nl (Dutch),     14: pl (Polish),     15: id (Indonesian),
 *  16: vi (Vietnamese),17: th (Thai),     18: ta (Tamil),      19: bn (Bengali),
 *  20: ur (Urdu).
 */
class MultilingualG2PN {
public:
    static constexpr int ARTICULATORY_DIM = 62;

    MultilingualG2PN();
    ~MultilingualG2PN() = default;

    // Normalizes numbers, currencies, contractions, punctuation in target language
    std::string normalizeText(const std::string& input_text, const std::string& lang_code) const;

    // 21-Language Colloquial Slang & Native Accent Reducer
    std::string applyColloquialSlang(const std::string& text, const std::string& lang_code) const;

    // Splits normalized text into streaming clauses (3-6 words or punctuation boundaries)
    std::vector<std::string> segmentClauses(const std::string& normalized_text, int max_words_per_clause = 5) const;

    // Splits text with intonation tagging (Question, Exclamation, Statement, Comma) & natural breath timing
    std::vector<ClauseSegment> segmentClausesWithIntonation(const std::string& raw_text, const std::string& lang_code) const;

    // Converts a single text clause to IPA phoneme IDs, 62-D Articulatory Vectors, and clause metadata
    PhonemeSequence processClause(const std::string& clause_text,
                                  const std::string& lang_code,
                                  ClauseType clause_type = ClauseType::STATEMENT) const;

    // Resolves language ISO code (e.g. "en", "hi", "es") to integer lang_id (0-20)
    int32_t getLanguageId(const std::string& lang_code) const;

    // Resolves language name / code from integer lang_id
    std::string getLanguageCode(int32_t lang_id) const;

    // Checks whether language is lexical-tonal (zh, th, vi)
    bool isTonalLanguage(int32_t lang_id) const noexcept;
    bool isTonalLanguage(const std::string& lang_code) const;

    // Total phoneme vocabulary size
    size_t getVocabSize() const noexcept { return m_ipa_to_id.size(); }

    // Map single IPA token to integer ID
    int32_t getPhonemeId(const std::string& ipa_symbol) const;

    // Articulatory 62-D Vector Lookup for IMS-Toucan
    const float* getArticulatoryVector(int32_t phoneme_id) const;
    std::vector<float> extractArticulatoryMatrix(const std::vector<int32_t>& phoneme_ids,
                                                 int32_t lang_id = 0) const;

    // Direct constexpr archetype access
    static constexpr const ArticulatoryLUT::TableType& getConstexprArchetypeTable() noexcept {
        return ArticulatoryLUT::kConstexprArticulatoryTable;
    }

    // Special Token IDs
    static constexpr int32_t TOKEN_PAD = 0;
    static constexpr int32_t TOKEN_BOS = 1;
    static constexpr int32_t TOKEN_EOS = 2;
    static constexpr int32_t TOKEN_PAUSE_SHORT = 3;   // ~100ms pause
    static constexpr int32_t TOKEN_PAUSE_MEDIUM = 4;  // ~180ms comma pause
    static constexpr int32_t TOKEN_PAUSE_LONG = 5;    // ~320ms sentence pause

private:
    void initVocab();
    void initLanguageMap();
    void initArticulatoryTable();

    // Normalizers
    std::string expandNumbers(const std::string& text, const std::string& lang) const;
    std::string expandCurrencies(const std::string& text, const std::string& lang) const;
    std::string expandContractions(const std::string& text, const std::string& lang) const;
    std::string cleanPunctuation(const std::string& text) const;

    // Language specific phonemizers (all 21 languages)
    std::vector<std::string> phonemizeEnglish(const std::string& text) const;
    std::vector<std::string> phonemizeSpanish(const std::string& text) const;
    std::vector<std::string> phonemizeFrench(const std::string& text) const;
    std::vector<std::string> phonemizeGerman(const std::string& text) const;
    std::vector<std::string> phonemizeIndic(const std::string& text, const std::string& lang) const;
    std::vector<std::string> phonemizeArabicUrdu(const std::string& text, const std::string& lang) const;
    std::vector<std::string> phonemizeChinese(const std::string& text) const;
    std::vector<std::string> phonemizeJapanese(const std::string& text) const;
    std::vector<std::string> phonemizeKorean(const std::string& text) const;
    std::vector<std::string> phonemizeRussian(const std::string& text) const;
    std::vector<std::string> phonemizeTonalSEAsian(const std::string& text, const std::string& lang) const;
    std::vector<std::string> phonemizeGeneralLatin(const std::string& text, const std::string& lang) const;

    std::unordered_map<std::string, int32_t> m_ipa_to_id;
    std::unordered_map<std::string, int32_t> m_lang_to_id;
    std::vector<std::string> m_id_to_lang;
    std::vector<std::vector<float>> m_articulatory_table; // [512, 62] derived from constexpr LUT + modifiers
};

} // namespace HDModularTTS
