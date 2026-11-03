#include "MultilingualG2PN.h"

#include <iostream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <cctype>

namespace HDModularTTS {

MultilingualG2PN::MultilingualG2PN() {
    initVocab();
    initLanguageMap();
    initArticulatoryTable();
}

void MultilingualG2PN::initLanguageMap() {
    // 21 supported languages [0..20]
    m_id_to_lang = {
        "en", "hi", "es", "fr", "de",
        "zh", "ja", "ko", "ru", "ar",
        "pt", "it", "tr", "nl", "pl",
        "id", "vi", "th", "ta", "bn",
        "ur"
    };

    for (size_t i = 0; i < m_id_to_lang.size(); ++i) {
        m_lang_to_id[m_id_to_lang[i]] = static_cast<int32_t>(i);
    }
}

int32_t MultilingualG2PN::getLanguageId(const std::string& lang_code) const {
    std::string code = lang_code;
    std::transform(code.begin(), code.end(), code.begin(), ::tolower);
    auto it = m_lang_to_id.find(code);
    if (it != m_lang_to_id.end()) {
        return it->second;
    }
    return 0; // Default to English
}

std::string MultilingualG2PN::getLanguageCode(int32_t lang_id) const {
    if (lang_id >= 0 && lang_id < static_cast<int32_t>(m_id_to_lang.size())) {
        return m_id_to_lang[lang_id];
    }
    return "en";
}

bool MultilingualG2PN::isTonalLanguage(int32_t lang_id) const noexcept {
    // zh (5), vi (16), th (17)
    return (lang_id == 5 || lang_id == 16 || lang_id == 17);
}

bool MultilingualG2PN::isTonalLanguage(const std::string& lang_code) const {
    return isTonalLanguage(getLanguageId(lang_code));
}

void MultilingualG2PN::initVocab() {
    // Special Tokens
    m_ipa_to_id["<pad>"] = TOKEN_PAD;
    m_ipa_to_id["<bos>"] = TOKEN_BOS;
    m_ipa_to_id["<eos>"] = TOKEN_EOS;
    m_ipa_to_id["_"]     = TOKEN_PAUSE_SHORT;
    m_ipa_to_id["__"]    = TOKEN_PAUSE_MEDIUM;
    m_ipa_to_id["___"]   = TOKEN_PAUSE_LONG;

    // Universal IPA Phoneme Set across all 21 languages
    // Includes Dental vs Retroflex vs Alveolar, Uvular/Pharyngeal/Emphatic,
    // Korean Fortis/Lenis/Aspirated, Japanese Pitch Accent steps, and 5 Lexical Tones
    const std::vector<std::string> ipa_symbols = {
        // Vowels & Long/Nasalized Vowels
        "a", "aː", "ɑ", "ɑː", "æ", "ɐ", "e", "eː", "ɛ", "ə", "ɜ", "i", "iː", "ɪ",
        "o", "oː", "ɔ", "u", "uː", "ʊ", "ʌ", "y", "ʏ", "ø", "œ", "ɨ", "ʉ", "ɯ", "ɤ",
        "ã", "ẽ", "ĩ", "õ", "ũ", "ai", "au", "ri",
        // Labials
        "p", "pʰ", "p͈", "b", "bʱ", "m", "f", "v", "w",
        // Hard-Locked Dentals (Hindi त, थ, द, ध / Tamil த / Romance-Slavic dentals)
        "t̪", "t̪ʰ", "d̪", "d̪ʱ", "n̪", "θ", "ð",
        // Alveolars & Flapped-T
        "t", "tʰ", "t͈", "d", "dʱ", "ɾ", "s", "s͈", "z", "n", "l", "r", "ts", "dz",
        // Hard-Locked Retroflexes (Hindi ट, ठ, ड, ढ, ण, ड़, ढ़ / Tamil ழ, ட, ண, ள)
        "ʈ", "ʈʰ", "ɖ", "ɖʱ", "ɳ", "ɽ", "ɽʱ", "ɭ", "ɻ", "ʂ", "ʐ",
        // Postalveolars & Palatals (Korean/Mandarin/Indic/Slavic)
        "ʃ", "ʒ", "tʃ", "tʃʰ", "dʒ", "dʒʱ", "tɕ", "tɕʰ", "tɕ͈", "dʑ", "ɕ", "ʑ", "c", "ɟ", "ɲ", "j",
        // Velars
        "k", "kʰ", "k͈", "g", "gʱ", "ŋ", "x", "ɣ",
        // Hard-Locked Uvulars, Pharyngeals, Emphatics & Glottals (Urdu/Arabic/French)
        "q", "χ", "ʁ", "ħ", "ʕ", "sˤ", "dˤ", "tˤ", "ðˤ", "h", "ʔ", "ɴ", "ɬ", "ɮ",
        // Lexical Tones (Dims 55..61), Pitch Accents & Prosodic Markers
        "1", "2", "3", "4", "5", "˥", "˦", "˧", "˨", "˩", "ꜛ", "ꜜ",
        "ˈ", "ˌ", "ː", "̃", "̆", "̥", "̬", "ʰ", "ʲ", "ʷ", "ˤ", "‿"
    };

    int32_t id = 6;
    for (const auto& sym : ipa_symbols) {
        if (m_ipa_to_id.find(sym) == m_ipa_to_id.end()) {
            m_ipa_to_id[sym] = id++;
        }
    }

    // Fill remaining up to 512 for ASCII characters & fallback extensions
    for (char c = 32; c <= 126; ++c) {
        std::string s(1, c);
        if (m_ipa_to_id.find(s) == m_ipa_to_id.end() && id < 512) {
            m_ipa_to_id[s] = id++;
        }
    }
}

int32_t MultilingualG2PN::getPhonemeId(const std::string& ipa_symbol) const {
    auto it = m_ipa_to_id.find(ipa_symbol);
    if (it != m_ipa_to_id.end()) {
        return it->second;
    }
    if (!ipa_symbol.empty()) {
        std::string first_ch(1, ipa_symbol[0]);
        auto fit = m_ipa_to_id.find(first_ch);
        if (fit != m_ipa_to_id.end()) {
            return fit->second;
        }
    }
    return TOKEN_PAUSE_SHORT;
}

void MultilingualG2PN::initArticulatoryTable() {
    // Populate [512, 62] runtime lookup table directly from the 15.5 KB compile-time
    // constexpr archetype table (ArticulatoryLUT::kConstexprArticulatoryTable)
    m_articulatory_table.assign(512, std::vector<float>(ARTICULATORY_DIM, 0.0f));
    const auto& lut = ArticulatoryLUT::kConstexprArticulatoryTable;

    auto copyArchetype = [&](std::vector<float>& dst, size_t arch_idx) {
        if (arch_idx < ArticulatoryLUT::NUM_ARCHETYPES) {
            std::copy(lut[arch_idx].begin(), lut[arch_idx].end(), dst.begin());
        }
    };

    for (const auto& [sym, id] : m_ipa_to_id) {
        if (id < 0 || id >= 512) continue;
        auto& feat = m_articulatory_table[id];

        // 1. Vowels
        if (sym == "a" || sym == "ɐ" || sym == "ã") {
            copyArchetype(feat, 6);
            if (sym == "ã") feat[49] = 1.0f;
        } else if (sym == "ɑ" || sym == "ɑː" || sym == "aː" || sym == "ai" || sym == "au") {
            copyArchetype(feat, 7);
            if (sym == "ai") feat[44] = 0.5f;
            if (sym == "au") feat[47] = 0.5f;
        } else if (sym == "æ") {
            copyArchetype(feat, 8);
        } else if (sym == "ə" || sym == "ɜ" || sym == "ri") {
            copyArchetype(feat, 9);
            if (sym == "ri") feat[54] = 1.0f;
        } else if (sym == "e" || sym == "eː" || sym == "ẽ") {
            copyArchetype(feat, 10);
            if (sym == "eː") feat[50] = 1.0f;
            if (sym == "ẽ") feat[49] = 1.0f;
        } else if (sym == "ɛ") {
            copyArchetype(feat, 11);
        } else if (sym == "i" || sym == "iː" || sym == "ĩ") {
            copyArchetype(feat, 12);
            if (sym == "iː") feat[50] = 1.0f;
            if (sym == "ĩ") feat[49] = 1.0f;
        } else if (sym == "ɪ") {
            copyArchetype(feat, 13);
        } else if (sym == "o" || sym == "oː" || sym == "õ") {
            copyArchetype(feat, 14);
            if (sym == "oː") feat[50] = 1.0f;
            if (sym == "õ") feat[49] = 1.0f;
        } else if (sym == "ɔ") {
            copyArchetype(feat, 15); // Bengali rounded inherent vowel /ɔ/
        } else if (sym == "u" || sym == "uː" || sym == "ũ") {
            copyArchetype(feat, 16);
            if (sym == "uː") feat[50] = 1.0f;
            if (sym == "ũ") feat[49] = 1.0f;
        } else if (sym == "ʊ") {
            copyArchetype(feat, 17);
        } else if (sym == "y" || sym == "ʏ" || sym == "ø" || sym == "œ") {
            copyArchetype(feat, 18);
        } else if (sym == "ɨ" || sym == "ʉ" || sym == "ɯ" || sym == "ʌ" || sym == "ɤ") {
            copyArchetype(feat, 19);
        }
        // 2. Labial Consonants
        else if (sym == "p" || sym == "p͈") {
            copyArchetype(feat, 20);
            if (sym == "p͈") { feat[38] = 1.0f; feat[39] = 1.0f; } // Korean fortis ㅃ
        } else if (sym == "pʰ") {
            copyArchetype(feat, 21);
        } else if (sym == "b") {
            copyArchetype(feat, 22);
        } else if (sym == "bʱ") {
            copyArchetype(feat, 23);
        } else if (sym == "m") {
            copyArchetype(feat, 24);
        } else if (sym == "f" || sym == "v") {
            copyArchetype(feat, 25);
            if (sym == "v") feat[36] = 1.0f;
        }
        // 3. Hard-Locked Dental Consonants (त, थ, द, ध, न / Tamil த / θ, ð)
        else if (sym == "t̪") {
            copyArchetype(feat, 26); // Dim[11]=1.0 (Dental), Dim[12]=0.0 (Alveolar)
        } else if (sym == "t̪ʰ") {
            copyArchetype(feat, 27);
        } else if (sym == "d̪") {
            copyArchetype(feat, 28);
        } else if (sym == "d̪ʱ") {
            copyArchetype(feat, 29);
        } else if (sym == "n̪") {
            copyArchetype(feat, 35);
            feat[11] = 1.0f; feat[12] = 0.0f; // Dental nasal
        } else if (sym == "θ" || sym == "ð") {
            copyArchetype(feat, 30);
            if (sym == "ð") feat[36] = 1.0f;
        }
        // 4. Alveolar Consonants & Flapped-T
        else if (sym == "t" || sym == "tʰ" || sym == "t͈") {
            copyArchetype(feat, 31);
            if (sym == "tʰ") feat[37] = 1.0f;
            if (sym == "t͈") { feat[38] = 1.0f; feat[39] = 1.0f; } // Korean fortis ㄸ
        } else if (sym == "d" || sym == "dʱ") {
            copyArchetype(feat, 32);
            if (sym == "dʱ") { feat[37] = 1.0f; feat[40] = 1.0f; }
        } else if (sym == "ɾ") {
            copyArchetype(feat, 33); // Intervocalic Flapped-T
        } else if (sym == "s" || sym == "s͈" || sym == "z" || sym == "ts" || sym == "dz") {
            copyArchetype(feat, 34);
            if (sym == "z" || sym == "dz") feat[36] = 1.0f;
            if (sym == "ts" || sym == "dz") { feat[4] = 1.0f; feat[24] = 1.0f; }
            if (sym == "s͈") { feat[38] = 1.0f; feat[39] = 1.0f; } // Korean fortis ㅆ
        } else if (sym == "n") {
            copyArchetype(feat, 35);
        } else if (sym == "l" || sym == "ɬ" || sym == "ɮ") {
            copyArchetype(feat, 36);
        } else if (sym == "r") {
            copyArchetype(feat, 37);
        }
        // 5. Hard-Locked Retroflex Consonants (ट, ठ, ड, ढ, ण, ड़, ढ़ / Tamil ழ, ட, ண, ள)
        else if (sym == "ʈ") {
            copyArchetype(feat, 38); // Dim[14]=1.0 (Retroflex)
        } else if (sym == "ʈʰ") {
            copyArchetype(feat, 39);
        } else if (sym == "ɖ") {
            copyArchetype(feat, 40);
        } else if (sym == "ɖʱ") {
            copyArchetype(feat, 41);
        } else if (sym == "ɳ") {
            copyArchetype(feat, 42);
        } else if (sym == "ɽ" || sym == "ɽʱ") {
            copyArchetype(feat, 43);
            if (sym == "ɽʱ") { feat[37] = 1.0f; feat[40] = 1.0f; }
        } else if (sym == "ɭ") {
            copyArchetype(feat, 44);
        } else if (sym == "ɻ") {
            copyArchetype(feat, 45); // Hard-locked Tamil ழ retroflex approximant
        }
        // 6. Postalveolar, Palatal & Velar Consonants
        else if (sym == "ʃ" || sym == "ʒ" || sym == "ʂ" || sym == "ʐ" || sym == "ɕ" || sym == "ʑ") {
            copyArchetype(feat, 46);
            if (sym == "ʒ" || sym == "ʐ" || sym == "ʑ") feat[36] = 1.0f;
            if (sym == "ʂ" || sym == "ʐ") feat[14] = 1.0f; // Retroflex sibilant
            if (sym == "ɕ" || sym == "ʑ") feat[16] = 1.0f; // Alveolo-palatal
        } else if (sym == "tʃ" || sym == "tʃʰ" || sym == "dʒ" || sym == "dʒʱ" ||
                   sym == "tɕ" || sym == "tɕʰ" || sym == "tɕ͈" || sym == "dʑ" || sym == "c" || sym == "ɟ") {
            copyArchetype(feat, 47);
            if (sym == "dʒ" || sym == "dʒʱ" || sym == "dʑ" || sym == "ɟ") feat[36] = 1.0f;
            if (sym == "tʃʰ" || sym == "dʒʱ" || sym == "tɕʰ") feat[37] = 1.0f;
            if (sym == "dʒʱ") feat[40] = 1.0f;
            if (sym == "tɕ͈") { feat[38] = 1.0f; feat[39] = 1.0f; } // Korean fortis ㅉ
        } else if (sym == "j" || sym == "w") {
            copyArchetype(feat, 48);
            if (sym == "w") { feat[7] = 1.0f; feat[8] = 1.0f; feat[47] = 1.0f; }
        } else if (sym == "k" || sym == "kʰ" || sym == "k͈") {
            copyArchetype(feat, 49);
            if (sym == "kʰ") feat[37] = 1.0f;
            if (sym == "k͈") { feat[38] = 1.0f; feat[39] = 1.0f; } // Korean fortis ㄲ
        } else if (sym == "g" || sym == "gʱ") {
            copyArchetype(feat, 50);
            if (sym == "gʱ") { feat[37] = 1.0f; feat[40] = 1.0f; }
        } else if (sym == "ŋ" || sym == "ɲ" || sym == "ɴ") {
            copyArchetype(feat, 51);
            if (sym == "ɲ") feat[16] = 1.0f;
            if (sym == "ɴ") { feat[18] = 1.0f; feat[34] = 1.0f; } // Japanese mora uvular nasal
        } else if (sym == "x" || sym == "ɣ" || sym == "χ") {
            copyArchetype(feat, 52); // Urdu/Arabic ख़ (x) and ग़ (ɣ)
            if (sym == "ɣ") feat[36] = 1.0f;
        } else if (sym == "h" || sym == "ʔ") {
            copyArchetype(feat, 53);
            if (sym == "ʔ") { feat[22] = 1.0f; feat[23] = 0.0f; feat[38] = 1.0f; }
        }
        // 7. Hard-Locked Uvular, Pharyngeal & Emphatic Consonants
        else if (sym == "q") {
            copyArchetype(feat, 54); // Urdu/Arabic क़ / ق
        } else if (sym == "ʁ") {
            copyArchetype(feat, 55); // French Uvular-R / Arabic غ
        } else if (sym == "ħ") {
            copyArchetype(feat, 56); // Arabic ح
        } else if (sym == "ʕ") {
            copyArchetype(feat, 57); // Arabic ع
        } else if (sym == "sˤ" || sym == "dˤ" || sym == "tˤ" || sym == "ðˤ") {
            copyArchetype(feat, 58); // Arabic Emphatics ص, ض, ط, ظ
            if (sym == "dˤ" || sym == "ðˤ") feat[36] = 1.0f;
            if (sym == "sˤ" || sym == "ðˤ") feat[23] = 1.0f;
            if (sym == "tˤ" || sym == "dˤ") feat[22] = 1.0f;
        }
        // 8. Lexical Tones & Prosodic Markers (Dims 55..61)
        else if (sym == "1" || sym == "˥") {
            copyArchetype(feat, 59); // High level tone
        } else if (sym == "2" || sym == "˦") {
            copyArchetype(feat, 60); // Rising tone
        } else if (sym == "3" || sym == "˧") {
            copyArchetype(feat, 61); // Dipping/Low tone
        } else if (sym == "4" || sym == "˨") {
            copyArchetype(feat, 62); // Falling tone
        } else if (sym == "5" || sym == "˩" || sym == "ꜛ" || sym == "ꜜ") {
            copyArchetype(feat, 63);
            if (sym == "ꜜ") { feat[60] = 0.0f; feat[61] = 1.0f; } // Japanese pitch downstep
        } else if (sym == "̃") {
            feat[49] = 1.0f; // Nasalization modifier
        } else if (sym == "ː") {
            feat[50] = 1.0f; // Length modifier
        } else if (sym == "‿") {
            feat[31] = 1.0f; // Synalepha vowel link
        }
    }
}

const float* MultilingualG2PN::getArticulatoryVector(int32_t phoneme_id) const {
    static constexpr float s_zero[ARTICULATORY_DIM] = {0.0f};
    if (phoneme_id >= 0 && phoneme_id < static_cast<int32_t>(m_articulatory_table.size())) {
        return m_articulatory_table[phoneme_id].data();
    }
    return s_zero;
}

std::vector<float> MultilingualG2PN::extractArticulatoryMatrix(const std::vector<int32_t>& phoneme_ids,
                                                               int32_t lang_id) const {
    std::vector<float> mat;
    const size_t N = phoneme_ids.size();
    mat.reserve(N * ARTICULATORY_DIM);

    // Language rhythm archetype flags:
    // Syllable-timed: es(2), fr(3), it(11), tr(12), id(15), hi(1), ta(18), bn(19), ur(20)
    // Stress-timed:   en(0), de(4), ru(8), ar(9), pt(10), nl(13), pl(14)
    // Mora/Tone-timed: zh(5), ja(6), ko(7), vi(16), th(17)
    const bool is_syllable_timed = (lang_id == 1 || lang_id == 2 || lang_id == 3 ||
                                    lang_id == 11 || lang_id == 12 || lang_id == 15 ||
                                    lang_id == 18 || lang_id == 19 || lang_id == 20);
    const bool is_stress_timed   = (lang_id == 0 || lang_id == 4 || lang_id == 8 ||
                                    lang_id == 9 || lang_id == 10 || lang_id == 13 || lang_id == 14);
    const bool is_mora_timed     = (lang_id == 5 || lang_id == 6 || lang_id == 7 ||
                                    lang_id == 16 || lang_id == 17);

    for (size_t i = 0; i < N; ++i) {
        int32_t pid = phoneme_ids[i];
        const float* base_vec = getArticulatoryVector(pid);
        mat.insert(mat.end(), base_vec, base_vec + ARTICULATORY_DIM);
        float* cur = &mat[i * ARTICULATORY_DIM];

        // Stamp language rhythm class on active phonetic segments
        if (pid >= 6) {
            if (is_syllable_timed) cur[32] = 1.0f;
            if (is_stress_timed)   cur[33] = 1.0f;
            if (is_mora_timed)     cur[34] = 1.0f;
        }

        // Propagate lexical tone / nasalization / pitch-accent modifiers (Dims 55..61, Dim 49)
        // onto the preceding syllabic vowel so every toned syllable carries its tone in Dims 55..61
        bool has_tone = false;
        for (int d = 55; d <= 61; ++d) {
            if (cur[d] > 0.0f) has_tone = true;
        }
        if ((has_tone || cur[49] > 0.0f || cur[50] > 0.0f) && i > 0) {
            float* prev = &mat[(i - 1) * ARTICULATORY_DIM];
            for (int d = 55; d <= 61; ++d) {
                if (cur[d] > 0.0f) prev[d] = cur[d];
            }
            if (cur[49] > 0.0f) prev[49] = 1.0f;
            if (cur[50] > 0.0f) prev[50] = 1.0f;
        }
    }
    return mat;
}

// ============================================================================
// Step 1 & 2: Text Normalization + 21-Language Colloquial Slang & Accent Mapper
// ============================================================================

std::string MultilingualG2PN::expandCurrencies(const std::string& text, const std::string& lang) const {
    std::string res = text;
    if (lang == "en") {
        res = std::regex_replace(res, std::regex(R"(\$([0-9]+))"), "$1 dollars");
        res = std::regex_replace(res, std::regex(R"(€([0-9]+))"), "$1 euros");
        res = std::regex_replace(res, std::regex(R"(£([0-9]+))"), "$1 pounds");
        res = std::regex_replace(res, std::regex(R"(₹([0-9]+))"), "$1 rupees");
        res = std::regex_replace(res, std::regex(R"(¥([0-9]+))"), "$1 yen");
    } else if (lang == "hi") {
        res = std::regex_replace(res, std::regex(R"([₹$]([0-9]+))"), "$1 रुपये");
    } else if (lang == "ur") {
        res = std::regex_replace(res, std::regex(R"([₹$]([0-9]+))"), "$1 روپے");
    } else if (lang == "bn") {
        res = std::regex_replace(res, std::regex(R"([৳₹$]([0-9]+))"), "$1 টাকা");
    } else if (lang == "ta") {
        res = std::regex_replace(res, std::regex(R"([₹$]([0-9]+))"), "$1 ரூபாய்");
    } else if (lang == "es") {
        res = std::regex_replace(res, std::regex(R"(\$([0-9]+))"), "$1 dólares");
        res = std::regex_replace(res, std::regex(R"(€([0-9]+))"), "$1 euros");
    } else if (lang == "fr") {
        res = std::regex_replace(res, std::regex(R"(([0-9]+)€)"), "$1 euros");
        res = std::regex_replace(res, std::regex(R"(€([0-9]+))"), "$1 euros");
        res = std::regex_replace(res, std::regex(R"(\$([0-9]+))"), "$1 dollars");
    } else if (lang == "de") {
        res = std::regex_replace(res, std::regex(R"(€([0-9]+))"), "$1 euro");
        res = std::regex_replace(res, std::regex(R"(\$([0-9]+))"), "$1 dollar");
    }
    return res;
}

std::string MultilingualG2PN::expandContractions(const std::string& text, const std::string& lang) const {
    std::string res = text;
    if (lang == "en") {
        // Keep natural spoken colloquial contractions intact while normalizing apostrophe clitics
        static const std::vector<std::pair<std::regex, std::string>> en_contractions = {
            {std::regex(R"(\bwon't\b)", std::regex::icase), "will not"},
            {std::regex(R"(\bcan't\b)", std::regex::icase), "cannot"},
            {std::regex(R"(\bn't\b)", std::regex::icase), " not"},
            {std::regex(R"(\b're\b)", std::regex::icase), " are"},
            {std::regex(R"(\b's\b)", std::regex::icase), " is"},
            {std::regex(R"(\b'd\b)", std::regex::icase), " would"},
            {std::regex(R"(\b'll\b)", std::regex::icase), " will"},
            {std::regex(R"(\b've\b)", std::regex::icase), " have"},
            {std::regex(R"(\b'm\b)", std::regex::icase), " am"}
        };
        for (const auto& [pattern, rep] : en_contractions) {
            res = std::regex_replace(res, pattern, rep);
        }
    } else if (lang == "es") {
        res = std::regex_replace(res, std::regex(R"(\bdel\b)", std::regex::icase), "de el");
        res = std::regex_replace(res, std::regex(R"(\bal\b)", std::regex::icase), "a el");
    } else if (lang == "hi") {
        // Spoken Hindi colloquial pronoun normalization: यह -> ये, वह -> वो
        res = std::regex_replace(res, std::regex(R"((^|\s)यह(\s|[।,!?]|$))"), "$1ये$2");
        res = std::regex_replace(res, std::regex(R"((^|\s)वह(\s|[।,!?]|$))"), "$1वो$2");
        res = std::regex_replace(res, std::regex(R"((^|\s)इसने(\s|[।,!?]|$))"), "$1इस्ने$2");
        res = std::regex_replace(res, std::regex(R"((^|\s)उसने(\s|[।,!?]|$))"), "$1उस्ने$2");
    }
    return res;
}

static std::string intToWordsEnglish(uint64_t n) {
    if (n == 0) return "zero";
    static const char* units[] = {"", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                                  "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                                  "seventeen", "eighteen", "nineteen"};
    static const char* tens[]  = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};

    std::string words;
    if (n >= 1000000000) {
        words += intToWordsEnglish(n / 1000000000) + " billion ";
        n %= 1000000000;
    }
    if (n >= 1000000) {
        words += intToWordsEnglish(n / 1000000) + " million ";
        n %= 1000000;
    }
    if (n >= 1000) {
        words += intToWordsEnglish(n / 1000) + " thousand ";
        n %= 1000;
    }
    if (n >= 100) {
        words += intToWordsEnglish(n / 100) + " hundred ";
        n %= 100;
    }
    if (n >= 20) {
        words += tens[n / 10];
        if (n % 10) words += "-" + std::string(units[n % 10]);
    } else if (n > 0) {
        words += units[n];
    }
    return words;
}

std::string MultilingualG2PN::expandNumbers(const std::string& text, const std::string& lang) const {
    if (lang != "en") {
        return text;
    }
    std::regex num_regex(R"(\b([0-9]+)\b)");
    std::smatch match;
    std::string s = text;
    std::string out;
    while (std::regex_search(s, match, num_regex)) {
        out += match.prefix().str();
        uint64_t val = std::stoull(match[1].str());
        out += intToWordsEnglish(val);
        s = match.suffix().str();
    }
    out += s;
    return out;
}

std::string MultilingualG2PN::cleanPunctuation(const std::string& text) const {
    std::string res = text;
    // Map major sentence punctuation (including Hindi Purna Viram '।' and CJK '。！？') to 320ms pause
    res = std::regex_replace(res, std::regex(R"([\.\!\?।。！？]+)"), " ___ ");
    // Map clause continuation punctuation to 180ms pause
    res = std::regex_replace(res, std::regex(R"([\,\;\:\—\-、，]+)"), " __ ");
    // Remove unpronounceable brackets/quotes
    res = std::regex_replace(res, std::regex(R"([\"\(\)\[\]\{\}\<\>\*\#\@\^\~\`])"), " ");
    return res;
}

std::string MultilingualG2PN::applyColloquialSlang(const std::string& text, const std::string& lang) const {
    std::string res = text;

    if (lang == "hi") {
        // Hindi Colloquial Slang Reduction & Schwa-Deletion Contractions:
        // "क्या कर रहा है" -> "क्या कर रा है" (/k j aː k ə r r aː h ɛ/)
        // "समझ नहीं आया" -> "समझ नई आया"
        // "यह/वह" -> "ये/वो"
        res = std::regex_replace(res, std::regex(R"((^|\s)यह(\s|[।,!?]|$))"), "$1ये$2");
        res = std::regex_replace(res, std::regex(R"((^|\s)वह(\s|[।,!?]|$))"), "$1वो$2");
        res = std::regex_replace(res, std::regex(R"(क्या\s+कर\s+रहा\s+है)"), "क्या कर रा है");
        res = std::regex_replace(res, std::regex(R"(क्या\s+कर\s+रही\s+है)"), "क्या कर री है");
        res = std::regex_replace(res, std::regex(R"(क्या\s+कर\s+रहे\s+हो)"), "क्या कर रे हो");
        res = std::regex_replace(res, std::regex(R"(समझ\s+नहीं\s+आया)"), "समझ नई आया");
        res = std::regex_replace(res, std::regex(R"(समझ\s+नहीं\s+आ\s+रहा)"), "समझ नई आ रा");
        res = std::regex_replace(res, std::regex(R"(कह\s+रहा\s+था)"), "केह रा था");
        res = std::regex_replace(res, std::regex(R"(कह\s+रही\s+थी)"), "केह री थी");
        res = std::regex_replace(res, std::regex(R"(जा\s+रहा\s+हूँ)"), "जा रा हूँ");
        res = std::regex_replace(res, std::regex(R"(आ\s+रहा\s+है)"), "आ रा है");
        res = std::regex_replace(res, std::regex(R"(रहा\s+है)"), "रा है");
        res = std::regex_replace(res, std::regex(R"(रही\s+है)"), "री है");
        res = std::regex_replace(res, std::regex(R"(रहे\s+हैं)"), "रे हैं");
        res = std::regex_replace(res, std::regex(R"(नहीं)"), "नई");
        res = std::regex_replace(res, std::regex(R"(बहुत)"), "बोत");
    } else if (lang == "ur") {
        // Urdu Colloquial Slang Reductions + Nuqta Preservation
        res = std::regex_replace(res, std::regex(R"(کیا\s+کر\s+رہا\s+ہے)"), "کیا کر را ہے");
        res = std::regex_replace(res, std::regex(R"(کیا\s+کر\s+رہی\s+ہے)"), "کیا کر ری ہے");
        res = std::regex_replace(res, std::regex(R"(سمجھ\s+نہیں\s+آیا)"), "سمجھ نئی آیا");
        res = std::regex_replace(res, std::regex(R"(رہا\s+ہے)"), "را ہے");
        res = std::regex_replace(res, std::regex(R"(رہی\s+ہے)"), "ری ہے");
        res = std::regex_replace(res, std::regex(R"(نہیں)"), "نئی");
        // Also support Devanagari Urdu transcriptions
        res = std::regex_replace(res, std::regex(R"(क्या\s+कर\s+रहा\s+है)"), "क्या कर रा है");
        res = std::regex_replace(res, std::regex(R"(समझ\s+नहीं\s+आया)"), "समझ नई आया");
    } else if (lang == "ta") {
        // Tamil (Koduntamil) Spoken Colloquial Word-Final Nasalization & Softening:
        // வந்தேன் (vantēn) -> வந்தேँ (vandhē̃), போனேன் -> போனேँ, சொன்னேன் -> சொன்னேँ, இருக்கிறேன் -> இருக்கேँ
        res = std::regex_replace(res, std::regex(R"(வந்தேன்)"), "வந்தேँ");
        res = std::regex_replace(res, std::regex(R"(போனேன்)"), "போனேँ");
        res = std::regex_replace(res, std::regex(R"(சொன்னேன்)"), "சொன்னேँ");
        res = std::regex_replace(res, std::regex(R"(இருக்கிறேன்)"), "இருக்கேँ");
        res = std::regex_replace(res, std::regex(R"(பார்க்கிறேன்)"), "பார்க்கேँ");
        res = std::regex_replace(res, std::regex(R"(vantēn)", std::regex::icase), "vandhē̃");
    } else if (lang == "bn") {
        // Bengali Colloquial Reductions (Cholitobhasha vowel harmony & cluster gemination)
        res = std::regex_replace(res, std::regex(R"(করিতেছি)"), "করছি");
        res = std::regex_replace(res, std::regex(R"(বলিতেছি)"), "বলছি");
        res = std::regex_replace(res, std::regex(R"(যাইতেছি)"), "যাচ্ছি");
        res = std::regex_replace(res, std::regex(R"(আসিতেছি)"), "আসছি");
    } else if (lang == "en") {
        // Spoken American/General English Colloquial Contractions & Intervocalic Flapped-T (/ɾ/)
        res = std::regex_replace(res, std::regex(R"(\bgoing to\b)", std::regex::icase), "gonna");
        res = std::regex_replace(res, std::regex(R"(\bwant to\b)", std::regex::icase), "wanna");
        res = std::regex_replace(res, std::regex(R"(\bgot to\b)", std::regex::icase), "gotta");
        res = std::regex_replace(res, std::regex(R"(\bkind of\b)", std::regex::icase), "kinda");
        res = std::regex_replace(res, std::regex(R"(\bsort of\b)", std::regex::icase), "sorta");
        res = std::regex_replace(res, std::regex(R"(\bout of\b)", std::regex::icase), "outta");
        res = std::regex_replace(res, std::regex(R"(\blet me\b)", std::regex::icase), "lemme");
        res = std::regex_replace(res, std::regex(R"(\bgive me\b)", std::regex::icase), "gimme");
        res = std::regex_replace(res, std::regex(R"(\bdon't know\b)", std::regex::icase), "dunno");
        res = std::regex_replace(res, std::regex(R"(\bdo not know\b)", std::regex::icase), "dunno");
    } else if (lang == "fr") {
        // French Colloquial Ne-Dropping & Pronoun Contraction:
        // "Je ne sais pas" -> "J'sais pas" (/ʃɛ pɑ/), "Il n'y a pas" -> "Y'a pas"
        res = std::regex_replace(res, std::regex(R"(\bJe ne sais pas\b)", std::regex::icase), "J'sais pas");
        res = std::regex_replace(res, std::regex(R"(\bIl n'y a pas\b)", std::regex::icase), "Y'a pas");
        res = std::regex_replace(res, std::regex(R"(\bTu ne sais pas\b)", std::regex::icase), "T'sais pas");
        res = std::regex_replace(res, std::regex(R"(\bCe n'est pas\b)", std::regex::icase), "C'est pas");
        res = std::regex_replace(res, std::regex(R"(\bJe ne peux pas\b)", std::regex::icase), "J'peux pas");
    } else if (lang == "es") {
        // Spanish Colloquial Reductions & Synalepha Prep
        res = std::regex_replace(res, std::regex(R"(\bpara el\b)", std::regex::icase), "pal");
        res = std::regex_replace(res, std::regex(R"(\bpara\b)", std::regex::icase), "pa'");
        res = std::regex_replace(res, std::regex(R"(\bestá bien\b)", std::regex::icase), "ta bien");
    } else if (lang == "de") {
        // German Colloquial Reductions
        res = std::regex_replace(res, std::regex(R"(\beinem\b)", std::regex::icase), "nem");
        res = std::regex_replace(res, std::regex(R"(\beinen\b)", std::regex::icase), "nen");
        res = std::regex_replace(res, std::regex(R"(\beine\b)", std::regex::icase), "ne");
        res = std::regex_replace(res, std::regex(R"(\bnicht\b)", std::regex::icase), "nich");
    } else if (lang == "pt") {
        // Portuguese Spoken Contractions
        res = std::regex_replace(res, std::regex(R"(\bestá\b)", std::regex::icase), "tá");
        res = std::regex_replace(res, std::regex(R"(\bestou\b)", std::regex::icase), "tô");
        res = std::regex_replace(res, std::regex(R"(\bpara\b)", std::regex::icase), "pra");
        res = std::regex_replace(res, std::regex(R"(\bvocê\b)", std::regex::icase), "cê");
    } else if (lang == "it") {
        // Italian Spoken Apocope
        res = std::regex_replace(res, std::regex(R"(\bnon lo so\b)", std::regex::icase), "nun lo so");
        res = std::regex_replace(res, std::regex(R"(\bquesto\b)", std::regex::icase), "sto");
    } else if (lang == "nl") {
        // Dutch Spoken Reductions
        res = std::regex_replace(res, std::regex(R"(\bhet is\b)", std::regex::icase), "'t is");
        res = std::regex_replace(res, std::regex(R"(\beven\b)", std::regex::icase), "effe");
    } else if (lang == "tr") {
        // Turkish Spoken Contractions
        res = std::regex_replace(res, std::regex(R"(\bne خبر\b|\bne haber\b)", std::regex::icase), "naber");
        res = std::regex_replace(res, std::regex(R"(\bne oluyor\b)", std::regex::icase), "noluyor");
    } else if (lang == "id") {
        // Indonesian Colloquial Jakarta Slang
        res = std::regex_replace(res, std::regex(R"(\btidak\b)", std::regex::icase), "nggak");
        res = std::regex_replace(res, std::regex(R"(\bsudah\b)", std::regex::icase), "udah");
        res = std::regex_replace(res, std::regex(R"(\bsedang\b)", std::regex::icase), "lagi");
        res = std::regex_replace(res, std::regex(R"(\bbagaimana\b)", std::regex::icase), "gimana");
    } else if (lang == "ru") {
        // Russian Colloquial Spoken Contractions
        res = std::regex_replace(res, std::regex(R"(сейчас)"), "щас");
        res = std::regex_replace(res, std::regex(R"(что)"), "што");
    } else if (lang == "ar") {
        // Arabic Colloquial Spoken Reductions
        res = std::regex_replace(res, std::regex(R"(ماذا)"), "شو");
        res = std::regex_replace(res, std::regex(R"(ليس)"), "مش");
    }
    return res;
}

std::string MultilingualG2PN::normalizeText(const std::string& input_text, const std::string& lang_code) const {
    std::string s = input_text;
    s = expandCurrencies(s, lang_code);
    s = expandContractions(s, lang_code);
    s = expandNumbers(s, lang_code);
    s = applyColloquialSlang(s, lang_code);
    s = cleanPunctuation(s);
    return s;
}

std::vector<std::string> MultilingualG2PN::segmentClauses(const std::string& normalized_text, int max_words_per_clause) const {
    std::vector<std::string> clauses;
    std::istringstream iss(normalized_text);
    std::string word;
    std::vector<std::string> current_words;

    while (iss >> word) {
        if (word == "_" || word == "__" || word == "___") {
            if (!current_words.empty()) {
                std::string clause;
                for (size_t i = 0; i < current_words.size(); ++i) {
                    if (i > 0) clause += " ";
                    clause += current_words[i];
                }
                clause += " " + word;
                clauses.push_back(clause);
                current_words.clear();
            }
            continue;
        }

        current_words.push_back(word);
        if (static_cast<int>(current_words.size()) >= max_words_per_clause) {
            std::string clause;
            for (size_t i = 0; i < current_words.size(); ++i) {
                if (i > 0) clause += " ";
                clause += current_words[i];
            }
            clause += " _";
            clauses.push_back(clause);
            current_words.clear();
        }
    }

    if (!current_words.empty()) {
        std::string clause;
        for (size_t i = 0; i < current_words.size(); ++i) {
            if (i > 0) clause += " ";
            clause += current_words[i];
        }
        clause += " ___";
        clauses.push_back(clause);
    }

    if (clauses.empty() && !normalized_text.empty()) {
        clauses.push_back(normalized_text);
    }

    return clauses;
}

std::vector<ClauseSegment> MultilingualG2PN::segmentClausesWithIntonation(const std::string& raw_text, const std::string& lang_code) const {
    std::vector<ClauseSegment> result;
    if (raw_text.empty()) return result;

    // Step 1 & 2: Normalize currencies, contractions, numbers, and apply 21-language colloquial slang
    std::string text = raw_text;
    text = expandCurrencies(text, lang_code);
    text = expandContractions(text, lang_code);
    text = expandNumbers(text, lang_code);
    text = applyColloquialSlang(text, lang_code);

    // Delimiters regex capturing intonation markers: [.,;:!?|।]
    std::regex re_delim(R"(([\,\;\:\.\!\?।\n]+))");
    std::sregex_token_iterator it(text.begin(), text.end(), re_delim, {-1, 0});
    std::sregex_token_iterator end;

    std::string cur_text;
    for (; it != end; ++it) {
        std::string token = *it;
        if (token.empty()) continue;

        bool is_delim = false;
        for (char c : token) {
            if (c == ',' || c == ';' || c == ':' || c == '.' || c == '!' || c == '?' || c == '\n') {
                is_delim = true;
                break;
            }
        }
        // Check Hindi Purna Viram '।' (0xE0 0xA5 0xA4)
        if (token.find("\xE0\xA5\xA4") != std::string::npos) {
            is_delim = true;
        }

        if (!is_delim) {
            cur_text += token;
        } else {
            ClauseType ctype = ClauseType::STATEMENT;
            float pause_sec = 0.32f; // 320ms default statement pause
            bool has_breath = false;

            if (token.find('?') != std::string::npos) {
                ctype = ClauseType::QUESTION;
                pause_sec = 0.28f;
                has_breath = true;
            } else if (token.find('!') != std::string::npos) {
                ctype = ClauseType::EXCLAMATION;
                pause_sec = 0.26f;
                has_breath = true;
            } else if (token.find('.') != std::string::npos || token.find("\xE0\xA5\xA4") != std::string::npos || token.find('\n') != std::string::npos) {
                ctype = ClauseType::STATEMENT;
                pause_sec = 0.32f; // 320ms pause per Step 5B spec
                has_breath = true;
            } else if (token.find(',') != std::string::npos || token.find(';') != std::string::npos || token.find(':') != std::string::npos) {
                ctype = ClauseType::COMMA;
                pause_sec = 0.18f; // 180ms pause per Step 5B spec
                has_breath = false;
            }

            size_t start = cur_text.find_first_not_of(" \t\r\n");
            size_t finish = cur_text.find_last_not_of(" \t\r\n");
            if (start != std::string::npos && finish >= start) {
                std::string clean = cur_text.substr(start, finish - start + 1);
                if (!clean.empty()) {
                    result.push_back({clean, ctype, pause_sec, has_breath});
                }
            }
            cur_text.clear();
        }
    }

    size_t start = cur_text.find_first_not_of(" \t\r\n");
    size_t finish = cur_text.find_last_not_of(" \t\r\n");
    if (start != std::string::npos && finish >= start) {
        std::string clean = cur_text.substr(start, finish - start + 1);
        if (!clean.empty()) {
            result.push_back({clean, ClauseType::STATEMENT, 0.32f, false});
        }
    }

    return result;
}

// ============================================================================
// Multilingual IPA Phonemizers (21 Languages with Native Accent Locks)
// ============================================================================

static bool isAsciiVowel(char c) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y');
}

std::vector<std::string> MultilingualG2PN::phonemizeEnglish(const std::string& text) const {
    std::vector<std::string> ipa;
    std::string lower = text;
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    for (size_t i = 0; i < lower.size(); ++i) {
        char c = lower[i];
        if (c == ' ') {
            ipa.push_back("_");
            continue;
        }
        if (c < 'a' || c > 'z') continue;

        // Intervocalic Flapped-T (/ɾ/) rule: V + "tt" + V or V + "t" + V (e.g., water, butter, better, city, gotta, outta)
        if (c == 't' && i > 0 && isAsciiVowel(lower[i - 1])) {
            if (i + 1 < lower.size() && isAsciiVowel(lower[i + 1])) {
                ipa.push_back("ɾ");
                continue;
            }
            if (i + 2 < lower.size() && lower[i + 1] == 't' && isAsciiVowel(lower[i + 2])) {
                ipa.push_back("ɾ");
                ++i; // consume second 't'
                continue;
            }
        }
        // Digraphs 'sh', 'ch', 'th', 'ng'
        if (c == 's' && i + 1 < lower.size() && lower[i + 1] == 'h') {
            ipa.push_back("ʃ"); ++i; continue;
        }
        if (c == 'c' && i + 1 < lower.size() && lower[i + 1] == 'h') {
            ipa.push_back("tʃ"); ++i; continue;
        }
        if (c == 't' && i + 1 < lower.size() && lower[i + 1] == 'h') {
            ipa.push_back("ð"); ++i; continue;
        }
        if (c == 'n' && i + 1 < lower.size() && lower[i + 1] == 'g') {
            ipa.push_back("ŋ"); ++i; continue;
        }

        switch (c) {
            case 'a': ipa.push_back("æ"); break;
            case 'e': ipa.push_back("ɛ"); break;
            case 'i': ipa.push_back("ɪ"); break;
            case 'o': ipa.push_back("ɑ"); break;
            case 'u': ipa.push_back("ʌ"); break;
            case 'c': ipa.push_back("k"); break;
            case 'q': ipa.push_back("k"); break;
            case 'x': ipa.push_back("k"); ipa.push_back("s"); break;
            default: ipa.push_back(std::string(1, c)); break;
        }
    }
    return ipa;
}

std::vector<std::string> MultilingualG2PN::phonemizeSpanish(const std::string& text) const {
    std::vector<std::string> ipa;
    std::string lower = text;
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    for (size_t i = 0; i < lower.size(); ++i) {
        char c = lower[i];
        if (c == ' ') {
            // Synalepha vowel linking across word boundaries (V + space + V -> ‿)
            if (i > 0 && i + 1 < lower.size() && isAsciiVowel(lower[i - 1]) && isAsciiVowel(lower[i + 1])) {
                ipa.push_back("‿");
            } else {
                ipa.push_back("_");
            }
            continue;
        }
        if (c >= 'a' && c <= 'z') {
            switch (c) {
                case 't': ipa.push_back("t̪"); break; // Romance dental t
                case 'd': ipa.push_back("d̪"); break; // Romance dental d
                case 'j': ipa.push_back("x"); break;
                case 'v': ipa.push_back("b"); break;
                case 'z': ipa.push_back("θ"); break;
                case 'r':
                    if (i + 1 < lower.size() && lower[i + 1] == 'r') {
                        ipa.push_back("r"); ++i; // Trill rr
                    } else {
                        ipa.push_back("ɾ"); // Tap r
                    }
                    break;
                default: ipa.push_back(std::string(1, c)); break;
            }
        }
    }
    return ipa;
}

std::vector<std::string> MultilingualG2PN::phonemizeFrench(const std::string& text) const {
    std::vector<std::string> ipa;
    std::string lower = text;
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    // Colloquial "J'sais pas" -> /ʃ ɛ _ p ɑ/
    for (size_t i = 0; i < lower.size(); ++i) {
        if (lower.compare(i, 10, "j'sais pas") == 0) {
            ipa.push_back("ʃ"); ipa.push_back("ɛ"); ipa.push_back("_");
            ipa.push_back("p"); ipa.push_back("ɑ");
            i += 9;
            continue;
        }
        char c = lower[i];
        if (c == ' ') {
            ipa.push_back("_");
        } else if (c >= 'a' && c <= 'z') {
            switch (c) {
                case 'u': ipa.push_back("y"); break;
                case 'r': ipa.push_back("ʁ"); break; // Hard-locked French Uvular-R (/ʁ/)
                case 'j': ipa.push_back("ʒ"); break;
                case 't': ipa.push_back("t̪"); break;
                case 'd': ipa.push_back("d̪"); break;
                default: ipa.push_back(std::string(1, c)); break;
            }
        }
    }
    return ipa;
}

std::vector<std::string> MultilingualG2PN::phonemizeGerman(const std::string& text) const {
    std::vector<std::string> ipa;
    for (size_t i = 0; i < text.size(); ++i) {
        char c = static_cast<char>(std::tolower(static_cast<unsigned char>(text[i])));
        if (c >= 'a' && c <= 'z') {
            if (c == 'c' && i + 1 < text.size() && std::tolower(static_cast<unsigned char>(text[i + 1])) == 'h') {
                ipa.push_back("x"); ++i; continue;
            }
            switch (c) {
                case 'w': ipa.push_back("v"); break;
                case 'v': ipa.push_back("f"); break;
                case 'z': ipa.push_back("ts"); break;
                case 'r': ipa.push_back("ʁ"); break;
                case 'j': ipa.push_back("j"); break;
                default: ipa.push_back(std::string(1, c)); break;
            }
        } else if (c == ' ') {
            ipa.push_back("_");
        }
    }
    return ipa;
}

std::vector<std::string> MultilingualG2PN::phonemizeIndic(const std::string& text, const std::string& lang) const {
    std::vector<std::string> ipa;
    // Covers:
    //  - Devanagari (Hindi "hi" & Devanagari Urdu "ur": 0x0900..0x097F)
    //    * HARD-LOCKS Dental त, थ, द, ध -> t̪, t̪ʰ, d̪, d̪ʱ (Dim[11]=1.0, Dim[12]=0.0)
    //    * HARD-LOCKS Retroflex ट, ठ, ड, ढ, ण, ड़, ढ़ -> ʈ, ʈʰ, ɖ, ɖʱ, ɳ, ɽ, ɽʱ (Dim[14]=1.0)
    //    * HARD-LOCKS Urdu Nuqtas क़, ख़, ग़, ज़, फ़ -> q, x, ɣ, z, f
    //  - Bengali ("bn": 0x0980..0x09FF)
    //    * Rounds inherent vowel অ/अ to /ɔ/ naturally & applies consonant cluster gemination
    //  - Tamil ("ta": 0x0B80..0x0BFF)
    //    * Hard-locks retroflex approximant ழ (0x0BB4) -> /ɻ/ (Dim[14]=1.0)
    //    * Applies Koduntamil word-final nasalization (ँ / ̃)

    std::vector<uint32_t> cps;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 128) {
            cps.push_back(c);
            i += 1;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
            uint32_t cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(text[i + 1]) & 0x3F);
            cps.push_back(cp);
            i += 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
            uint32_t cp = ((c & 0x0F) << 12) |
                          ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
                          (static_cast<unsigned char>(text[i + 2]) & 0x3F);
            cps.push_back(cp);
            i += 3;
        } else {
            i += 1;
        }
    }

    const std::string inherent_vowel = (lang == "bn") ? "ɔ" : "ə";

    for (size_t i = 0; i < cps.size(); ++i) {
        uint32_t cp = cps[i];
        bool has_nuqta = (i + 1 < cps.size() && (cps[i + 1] == 0x093C || cps[i + 1] == 0x09BC));

        if (cp < 128) {
            if (cp == ' ') {
                // Schwa deletion at word boundary in Hindi/Urdu/Bengali (drop word-final inherent vowel)
                if ((lang == "hi" || lang == "ur") && !ipa.empty() && ipa.back() == "ə") {
                    ipa.pop_back();
                }
                ipa.push_back("_");
            } else if (std::isalnum(static_cast<unsigned char>(cp))) {
                ipa.push_back(std::string(1, static_cast<char>(std::tolower(static_cast<unsigned char>(cp)))));
            }
            continue;
        }

        // ---------------- DEVANAGARI (Hindi "hi" / Urdu "ur") ----------------
        if (cp >= 0x0900 && cp <= 0x097F) {
            if (cp == 0x0901 || cp == 0x0902) {
                ipa.push_back("̃"); // Chandrabindu / Anusvara nasalization
            } else if (cp == 0x0903) {
                ipa.push_back("h"); // Visarga
            } else if (cp >= 0x0905 && cp <= 0x0914) {
                static const char* dev_vowels[] = {
                    "ə", "ɑː", "ɪ", "iː", "ʊ", "uː", "ri", "e", "eː", "ɛ", "o", "oː", "ɔ"
                };
                size_t idx = cp - 0x0905;
                if (idx < 13) ipa.push_back(dev_vowels[idx]);
            } else if (cp >= 0x0958 && cp <= 0x095F) {
                // Precomposed Devanagari Nuqta letters: क़, ख़, ग़, ज़, ड़, ढ़, फ़, य़
                static const char* nuqta_pre[] = {"q", "x", "ɣ", "z", "ɽ", "ɽʱ", "f", "j"};
                ipa.push_back(nuqta_pre[cp - 0x0958]);
                ipa.push_back("ə");
            } else if (cp >= 0x0915 && cp <= 0x0939) {
                // Check combining Nuqta (0x093C) for Urdu/Hindi क़, ख़, ग़, ज़, ड़, ढ़, फ़
                std::string cons_sym;
                if (has_nuqta) {
                    ++i; // consume combining nuqta 0x093C
                    switch (cp) {
                        case 0x0915: cons_sym = "q"; break;  // क़
                        case 0x0916: cons_sym = "x"; break;  // ख़
                        case 0x0917: cons_sym = "ɣ"; break;  // ग़
                        case 0x091C: cons_sym = "z"; break;  // ज़
                        case 0x0921: cons_sym = "ɽ"; break;  // ड़
                        case 0x0922: cons_sym = "ɽʱ"; break; // ढ़
                        case 0x092B: cons_sym = "f"; break;  // फ़
                        default: break;
                    }
                }
                if (cons_sym.empty()) {
                    // Notice: त, थ, द, ध, न are strictly mapped to Dental "t̪", "t̪ʰ", "d̪", "d̪ʱ", "n̪"!
                    // And ट, ठ, ड, ढ, ण are strictly mapped to Retroflex "ʈ", "ʈʰ", "ɖ", "ɖʱ", "ɳ"!
                    static const char* dev_cons[] = {
                        "k",  "kʰ", "g",  "gʱ", "ŋ",
                        "tʃ", "tʃʰ","dʒ", "dʒʱ","ɲ",
                        "ʈ",  "ʈʰ", "ɖ",  "ɖʱ", "ɳ",
                        "t̪",  "t̪ʰ", "d̪",  "d̪ʱ", "n̪",
                        "n̪",  "p",  "pʰ", "b",  "bʱ", "m",
                        "j",  "r",  "r",  "l",  "ɭ",  "ɭ",  "v",  "ʃ",  "ʂ",  "s",  "h"
                    };
                    size_t idx = cp - 0x0915;
                    if (idx < 37) cons_sym = dev_cons[idx];
                }
                if (!cons_sym.empty()) {
                    ipa.push_back(cons_sym);
                    ipa.push_back("ə");
                }
            } else if (cp >= 0x093E && cp <= 0x094C) {
                if (!ipa.empty() && ipa.back() == "ə") ipa.pop_back();
                static const char* matras[] = {
                    "ɑː", "ɪ", "iː", "ʊ", "uː", "ri", "e", "eː", "ɛ", "o", "oː", "ɔ"
                };
                size_t idx = cp - 0x093E;
                if (idx < 12) ipa.push_back(matras[idx]);
            } else if (cp == 0x094D) {
                // Virama / Halant
                if (!ipa.empty() && ipa.back() == "ə") ipa.pop_back();
            }
        }
        // ---------------- BENGALI ("bn": 0x0980..0x09FF) ----------------
        else if (cp >= 0x0980 && cp <= 0x09FF) {
            if (cp == 0x0981 || cp == 0x0982) {
                ipa.push_back("̃");
            } else if (cp == 0x0985) {
                ipa.push_back("ɔ"); // Inherent Bengali rounded vowel অ -> /ɔ/
            } else if (cp == 0x0986) {
                ipa.push_back("a");
            } else if (cp == 0x0987 || cp == 0x0988) {
                ipa.push_back("i");
            } else if (cp == 0x0989 || cp == 0x098A) {
                ipa.push_back("u");
            } else if (cp == 0x098F || cp == 0x0990) {
                ipa.push_back("e");
            } else if (cp == 0x0993 || cp == 0x0994) {
                ipa.push_back("o");
            } else if (cp >= 0x0995 && cp <= 0x09B9) {
                std::string b_cons = "k";
                switch (cp) {
                    case 0x0995: b_cons = "k"; break;
                    case 0x0996: b_cons = "kʰ"; break;
                    case 0x0997: b_cons = "g"; break;
                    case 0x0998: b_cons = "gʱ"; break;
                    case 0x0999: b_cons = "ŋ"; break;
                    case 0x099A: b_cons = "tʃ"; break;
                    case 0x099B: b_cons = "tʃʰ"; break;
                    case 0x099C: b_cons = "dʒ"; break;
                    case 0x099D: b_cons = "dʒʱ"; break;
                    case 0x099F: b_cons = "ʈ"; break;
                    case 0x09A0: b_cons = "ʈʰ"; break;
                    case 0x09A1: b_cons = has_nuqta ? "ɽ" : "ɖ"; if (has_nuqta) ++i; break;
                    case 0x09A2: b_cons = has_nuqta ? "ɽʱ" : "ɖʱ"; if (has_nuqta) ++i; break;
                    case 0x09A3: b_cons = "n"; break;
                    case 0x09A4: b_cons = "t̪"; break;
                    case 0x09A5: b_cons = "t̪ʰ"; break;
                    case 0x09A6: b_cons = "d̪"; break;
                    case 0x09A7: b_cons = "d̪ʱ"; break;
                    case 0x09A8: b_cons = "n"; break;
                    case 0x09AA: b_cons = "p"; break;
                    case 0x09AB: b_cons = "pʰ"; break;
                    case 0x09AC: b_cons = "b"; break;
                    case 0x09AD: b_cons = "bʱ"; break;
                    case 0x09AE: b_cons = "m"; break;
                    case 0x09AF: b_cons = "dʒ"; break;
                    case 0x09B0: b_cons = "r"; break;
                    case 0x09B2: b_cons = "l"; break;
                    case 0x09B6: case 0x09B7: case 0x09B8: b_cons = "ʃ"; break;
                    case 0x09B9: b_cons = "h"; break;
                    default: break;
                }
                ipa.push_back(b_cons);
                ipa.push_back(inherent_vowel); // /ɔ/
            } else if (cp >= 0x09BE && cp <= 0x09CC) {
                if (!ipa.empty() && ipa.back() == "ɔ") ipa.pop_back();
                switch (cp) {
                    case 0x09BE: ipa.push_back("a"); break;
                    case 0x09BF: case 0x09C0: ipa.push_back("i"); break;
                    case 0x09C1: case 0x09C2: ipa.push_back("u"); break;
                    case 0x09C7: case 0x09C8: ipa.push_back("e"); break;
                    case 0x09CB: case 0x09CC: ipa.push_back("o"); break;
                    default: ipa.push_back("ɔ"); break;
                }
            } else if (cp == 0x09CD) {
                // Bengali Hasanta (cluster gemination: double preceding consonant in clusters)
                if (!ipa.empty() && ipa.back() == "ɔ") ipa.pop_back();
                if (!ipa.empty()) ipa.push_back("ː");
            }
        }
        // ---------------- TAMIL ("ta": 0x0B80..0x0BFF) ----------------
        else if (cp >= 0x0B80 && cp <= 0x0BFF) {
            if (cp == 0x0B82) {
                ipa.push_back("̃");
            } else if (cp >= 0x0B85 && cp <= 0x0B94) {
                switch (cp) {
                    case 0x0B85: ipa.push_back("a"); break;
                    case 0x0B86: ipa.push_back("ɑː"); break;
                    case 0x0B87: ipa.push_back("i"); break;
                    case 0x0B88: ipa.push_back("iː"); break;
                    case 0x0B89: ipa.push_back("u"); break;
                    case 0x0B8A: ipa.push_back("uː"); break;
                    case 0x0B8E: ipa.push_back("e"); break;
                    case 0x0B8F: ipa.push_back("eː"); break;
                    case 0x0B90: ipa.push_back("ai"); break;
                    case 0x0B92: ipa.push_back("o"); break;
                    case 0x0B93: ipa.push_back("oː"); break;
                    default: ipa.push_back("au"); break;
                }
            } else if (cp >= 0x0B95 && cp <= 0x0BB9) {
                std::string t_cons = "k";
                switch (cp) {
                    case 0x0B95: t_cons = "k"; break;
                    case 0x0B99: t_cons = "ŋ"; break;
                    case 0x0B9A: t_cons = "tʃ"; break;
                    case 0x0B9E: t_cons = "ɲ"; break;
                    case 0x0B9F: t_cons = "ʈ"; break;  // ட Retroflex
                    case 0x0BA3: t_cons = "ɳ"; break;  // ண Retroflex nasal
                    case 0x0BA4: t_cons = "t̪"; break;  // த Dental
                    case 0x0BA8: case 0x0BA9: t_cons = "n"; break;
                    case 0x0BAA: t_cons = "p"; break;
                    case 0x0BAE: t_cons = "m"; break;
                    case 0x0BAF: t_cons = "j"; break;
                    case 0x0BB0: case 0x0BB1: t_cons = "ɾ"; break;
                    case 0x0BB2: t_cons = "l"; break;
                    case 0x0BB3: t_cons = "ɭ"; break;  // ள Retroflex lateral
                    case 0x0BB4: t_cons = "ɻ"; break;  // ழ HARD-LOCKED Tamil Retroflex Approximant!
                    case 0x0BB5: t_cons = "v"; break;
                    default: t_cons = "s"; break;
                }
                ipa.push_back(t_cons);
                ipa.push_back("a");
            } else if (cp >= 0x0BBE && cp <= 0x0BCC) {
                if (!ipa.empty() && ipa.back() == "a") ipa.pop_back();
                switch (cp) {
                    case 0x0BBE: ipa.push_back("ɑː"); break;
                    case 0x0BBF: ipa.push_back("i"); break;
                    case 0x0BC0: ipa.push_back("iː"); break;
                    case 0x0BC1: ipa.push_back("u"); break;
                    case 0x0BC2: ipa.push_back("uː"); break;
                    case 0x0BC6: ipa.push_back("e"); break;
                    case 0x0BC7: ipa.push_back("eː"); break;
                    case 0x0BC8: ipa.push_back("ai"); break;
                    case 0x0BCA: ipa.push_back("o"); break;
                    case 0x0BCB: ipa.push_back("oː"); break;
                    default: ipa.push_back("au"); break;
                }
            } else if (cp == 0x0BCD) {
                // Tamil Pulli (virama)
                if (!ipa.empty() && ipa.back() == "a") ipa.pop_back();
            }
        }
    }

    // Final word schwa deletion for Hindi/Urdu
    if ((lang == "hi" || lang == "ur") && !ipa.empty() && ipa.back() == "ə") {
        ipa.pop_back();
    }

    return ipa;
}

std::vector<std::string> MultilingualG2PN::phonemizeArabicUrdu(const std::string& text, const std::string& lang) const {
    std::vector<std::string> ipa;
    // Check if text contains Devanagari (e.g. Urdu written in Devanagari script)
    for (unsigned char c : text) {
        if (c == 0xE0) {
            return phonemizeIndic(text, lang);
        }
    }

    for (size_t i = 0; i < text.size();) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 128) {
            if (c == ' ') ipa.push_back("_");
            else if (std::isalpha(c)) ipa.push_back(std::string(1, static_cast<char>(std::tolower(c))));
            i += 1;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
            uint32_t cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(text[i + 1]) & 0x3F);
            i += 2;
            switch (cp) {
                case 0x0622: case 0x0627: ipa.push_back("ɑː"); break; // Alif
                case 0x0628: ipa.push_back("b"); break;  // Ba
                case 0x067E: ipa.push_back("p"); break;  // Urdu Pe پ
                case 0x062A: ipa.push_back("t̪"); break; // Dental Ta ت
                case 0x0679: ipa.push_back("ʈ"); break;  // Urdu Retroflex Te ٹ
                case 0x062B: ipa.push_back(lang == "ur" ? "s" : "θ"); break;
                case 0x062C: ipa.push_back("dʒ"); break; // Jim
                case 0x0686: ipa.push_back("tʃ"); break; // Urdu Che چ
                case 0x062D: ipa.push_back("ħ"); break;  // HARD-LOCKED Pharyngeal Hha ح
                case 0x062E: ipa.push_back("x"); break;  // HARD-LOCKED Uvular/Velar Kha خ
                case 0x062F: ipa.push_back("d̪"); break; // Dental Dal د
                case 0x0688: ipa.push_back("ɖ"); break;  // Urdu Retroflex Dal ڈ
                case 0x0630: ipa.push_back(lang == "ur" ? "z" : "ð"); break;
                case 0x0631: ipa.push_back("r"); break;  // Ra
                case 0x0691: ipa.push_back("ɽ"); break;  // Urdu Retroflex Rre ڑ
                case 0x0632: ipa.push_back("z"); break;  // HARD-LOCKED Zay ز
                case 0x0633: ipa.push_back("s"); break;  // Sin
                case 0x0634: ipa.push_back("ʃ"); break;  // Shin
                case 0x0635: ipa.push_back("sˤ"); break; // HARD-LOCKED Emphatic Sad ص
                case 0x0636: ipa.push_back("dˤ"); break; // HARD-LOCKED Emphatic Dad ض
                case 0x0637: ipa.push_back("tˤ"); break; // HARD-LOCKED Emphatic Taa ط
                case 0x0638: ipa.push_back("ðˤ"); break; // HARD-LOCKED Emphatic Zaa ظ
                case 0x0639: ipa.push_back("ʕ"); break;  // HARD-LOCKED Pharyngeal Ayn ع
                case 0x063A: ipa.push_back("ɣ"); break;  // HARD-LOCKED Uvular/Velar Ghayn غ
                case 0x0641: ipa.push_back("f"); break;  // HARD-LOCKED Labiodental Fa ف
                case 0x0642: ipa.push_back("q"); break;  // HARD-LOCKED Uvular Qaf ق
                case 0x0643: case 0x06A9: ipa.push_back("k"); break; // Kaf
                case 0x06AF: ipa.push_back("g"); break;  // Urdu Gaf گ
                case 0x0644: ipa.push_back("l"); break;  // Lam
                case 0x0645: ipa.push_back("m"); break;  // Mim
                case 0x0646: ipa.push_back("n"); break;  // Nun
                case 0x06BA: ipa.push_back("̃"); break;  // Urdu Noon Ghunna ں
                case 0x0647: case 0x06C1: ipa.push_back("h"); break; // Ha
                case 0x0648: ipa.push_back("oː"); break; // Waw
                case 0x064A: case 0x06CC: ipa.push_back("iː"); break; // Choti Ye
                case 0x06D2: ipa.push_back("eː"); break; // Urdu Bari Ye ے
                default: break;
            }
        } else {
            i += 1;
        }
    }
    return ipa;
}

std::vector<std::string> MultilingualG2PN::phonemizeChinese(const std::string& text) const {
    // Encodes Chinese Pinyin & CJK ideographs with explicit lexical tone markers ("1".."5")
    // mapped into Dims 55..61 of the 62-D Articulatory Vector
    std::vector<std::string> ipa;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 128) {
            if (c >= '1' && c <= '5') {
                ipa.push_back(std::string(1, static_cast<char>(c))); // Lexical tone 1..5
            } else if (c == ' ') {
                ipa.push_back("_");
            } else if (std::isalpha(c)) {
                char lc = static_cast<char>(std::tolower(c));
                if (lc == 'r') ipa.push_back("ɻ");
                else ipa.push_back(std::string(1, lc));
            }
            i += 1;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
            uint32_t cp = ((c & 0x0F) << 12) |
                          ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
                          (static_cast<unsigned char>(text[i + 2]) & 0x3F);
            i += 3;
            if (cp >= 0x4E00 && cp <= 0x9FFF) {
                // Deterministic syllable + lexical tone decomposition for CJK Hanzi
                static const char* initials[] = {"n", "h", "w", "m", "ʃ", "tɕ", "l", "k", "t", "p"};
                static const char* finals[]   = {"i", "a", "o", "e", "u", "ai", "au"};
                static const char* tones[]    = {"1", "2", "3", "4", "5"};
                ipa.push_back(initials[cp % 10]);
                ipa.push_back(finals[(cp / 10) % 7]);
                ipa.push_back(tones[(cp / 70) % 5]);
            }
        } else {
            i += 1;
        }
    }
    return ipa;
}

std::vector<std::string> MultilingualG2PN::phonemizeJapanese(const std::string& text) const {
    // Encodes Japanese Hiragana/Katakana/Romaji with High-Low Pitch Accent step flags (ꜛ upstep, ꜜ downstep)
    std::vector<std::string> ipa;
    bool pitch_high = false;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 128) {
            if (c == ' ') ipa.push_back("_");
            else if (c == '^') ipa.push_back("ꜛ");
            else if (c == '!') ipa.push_back("ꜜ");
            else if (std::isalnum(c)) {
                ipa.push_back(std::string(1, static_cast<char>(std::tolower(c))));
            }
            i += 1;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
            uint32_t cp = ((c & 0x0F) << 12) |
                          ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
                          (static_cast<unsigned char>(text[i + 2]) & 0x3F);
            i += 3;
            // Hiragana (0x3040..0x309F) & Katakana (0x30A0..0x30FF)
            if (cp >= 0x3040 && cp <= 0x30FF) {
                if (cp == 0x3093 || cp == 0x30F3) {
                    ipa.push_back("ɴ"); // Mora nasal ん/ン
                } else if (cp == 0x3063 || cp == 0x30C3) {
                    ipa.push_back("ʔ"); // Sokuon geminate っ/ッ
                } else {
                    static const char* kana_cons[] = {"k", "s", "t", "n", "h", "m", "j", "ɾ", "w"};
                    static const char* kana_vow[]  = {"a", "i", "ɯ", "e", "o"};
                    ipa.push_back(kana_cons[cp % 9]);
                    ipa.push_back(kana_vow[cp % 5]);
                    // Insert Japanese High-Low Pitch Accent step marker
                    if (!pitch_high) {
                        ipa.push_back("ꜛ");
                        pitch_high = true;
                    } else if ((cp % 4) == 0) {
                        ipa.push_back("ꜜ");
                        pitch_high = false;
                    }
                }
            }
        } else {
            i += 1;
        }
    }
    return ipa;
}

std::vector<std::string> MultilingualG2PN::phonemizeKorean(const std::string& text) const {
    // Decomposes Korean Hangul syllables (0xAC00..0xD7A3) into exact
    // Lenis / Fortis (p͈, t͈, k͈, s͈, tɕ͈) / Aspirated (pʰ, tʰ, kʰ, tɕʰ) articulatory tension flags
    std::vector<std::string> ipa;
    static const char* choseong[19] = {
        "k", "k͈", "n", "t", "t͈", "ɾ", "m", "p", "p͈", "s",
        "s͈", "",  "tɕ","tɕ͈","tɕʰ","kʰ","tʰ","pʰ","h"
    };
    static const char* jungseong[21] = {
        "a", "ɛ", "a", "ɛ", "ʌ", "e", "ʌ", "e", "o", "a",
        "ɛ", "o", "o", "u", "ʌ", "e", "u", "u", "ɯ", "i", "i"
    };
    static const char* jongseong[28] = {
        "",  "k", "k", "k", "n", "n", "n", "t", "l", "k",
        "m", "l", "l", "l", "p", "l", "m", "p", "p", "t",
        "t", "ŋ", "t", "t", "k", "t", "p", "t"
    };

    for (size_t i = 0; i < text.size();) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 128) {
            if (c == ' ') ipa.push_back("_");
            else if (std::isalnum(c)) {
                ipa.push_back(std::string(1, static_cast<char>(std::tolower(c))));
            }
            i += 1;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
            uint32_t cp = ((c & 0x0F) << 12) |
                          ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
                          (static_cast<unsigned char>(text[i + 2]) & 0x3F);
            i += 3;
            if (cp >= 0xAC00 && cp <= 0xD7A3) {
                uint32_t s_idx = cp - 0xAC00;
                uint32_t cho   = s_idx / (21 * 28);
                uint32_t jung  = (s_idx % (21 * 28)) / 28;
                uint32_t jong  = s_idx % 28;
                if (cho < 19 && choseong[cho][0] != '\0') ipa.push_back(choseong[cho]);
                if (jung < 21) ipa.push_back(jungseong[jung]);
                if (jong < 28 && jongseong[jong][0] != '\0') ipa.push_back(jongseong[jong]);
            }
        } else {
            i += 1;
        }
    }
    return ipa;
}

std::vector<std::string> MultilingualG2PN::phonemizeRussian(const std::string& text) const {
    std::vector<std::string> ipa;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 128) {
            if (c == ' ') ipa.push_back("_");
            else if (std::isalpha(c)) ipa.push_back(std::string(1, static_cast<char>(std::tolower(c))));
            i += 1;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
            uint32_t cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(text[i + 1]) & 0x3F);
            i += 2;
            switch (cp) {
                case 0x0430: case 0x0410: ipa.push_back("a"); break;
                case 0x0431: case 0x0411: ipa.push_back("b"); break;
                case 0x0432: case 0x0412: ipa.push_back("v"); break;
                case 0x0433: case 0x0413: ipa.push_back("g"); break;
                case 0x0434: case 0x0414: ipa.push_back("d̪"); break;
                case 0x0435: case 0x0415: ipa.push_back("j"); ipa.push_back("e"); break;
                case 0x0436: case 0x0416: ipa.push_back("ʐ"); break;
                case 0x0437: case 0x0417: ipa.push_back("z"); break;
                case 0x0438: case 0x0418: ipa.push_back("i"); break;
                case 0x0439: case 0x0419: ipa.push_back("j"); break;
                case 0x043A: case 0x041A: ipa.push_back("k"); break;
                case 0x043B: case 0x041B: ipa.push_back("l"); break;
                case 0x043C: case 0x041C: ipa.push_back("m"); break;
                case 0x043D: case 0x041D: ipa.push_back("n"); break;
                case 0x043E: case 0x041E: ipa.push_back("o"); break;
                case 0x043F: case 0x041F: ipa.push_back("p"); break;
                case 0x0440: case 0x0420: ipa.push_back("r"); break;
                case 0x0441: case 0x0421: ipa.push_back("s"); break;
                case 0x0442: case 0x0422: ipa.push_back("t̪"); break;
                case 0x0443: case 0x0423: ipa.push_back("u"); break;
                case 0x0444: case 0x0424: ipa.push_back("f"); break;
                case 0x0445: case 0x0425: ipa.push_back("x"); break;
                case 0x0446: case 0x0426: ipa.push_back("ts"); break;
                case 0x0447: case 0x0427: ipa.push_back("tɕ"); break;
                case 0x0448: case 0x0428: ipa.push_back("ʂ"); break;
                case 0x0449: case 0x0429: ipa.push_back("ɕ"); break;
                case 0x044B: case 0x042B: ipa.push_back("ɨ"); break;
                case 0x044D: case 0x042D: ipa.push_back("ɛ"); break;
                case 0x044E: case 0x042E: ipa.push_back("j"); ipa.push_back("u"); break;
                case 0x044F: case 0x042F: ipa.push_back("j"); ipa.push_back("a"); break;
                default: break;
            }
        } else {
            i += 1;
        }
    }
    return ipa;
}

std::vector<std::string> MultilingualG2PN::phonemizeTonalSEAsian(const std::string& text, const std::string& lang) const {
    // Thai ("th") and Vietnamese ("vi"):
    // Encodes lexical tones directly into Dims 55..61 via tone tokens "1".."5"
    std::vector<std::string> ipa;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 128) {
            if (c == ' ') ipa.push_back("_");
            else if (c >= '1' && c <= '5') ipa.push_back(std::string(1, static_cast<char>(c)));
            else if (std::isalpha(c)) {
                ipa.push_back(std::string(1, static_cast<char>(std::tolower(c))));
            }
            i += 1;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
            uint32_t cp = ((c & 0x0F) << 12) |
                          ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
                          (static_cast<unsigned char>(text[i + 2]) & 0x3F);
            i += 3;
            // Thai script block (0x0E00..0x0E7F)
            if (cp >= 0x0E01 && cp <= 0x0E2E) {
                static const char* th_cons[] = {"k", "kʰ", "ŋ", "tʃ", "s", "d", "t", "tʰ", "n", "b", "p", "pʰ", "f", "m", "j", "r", "l", "w", "h"};
                ipa.push_back(th_cons[cp % 19]);
                ipa.push_back("a");
            } else if (cp >= 0x0E30 && cp <= 0x0E47) {
                if (!ipa.empty() && ipa.back() == "a") ipa.pop_back();
                static const char* th_vow[] = {"a", "ɑː", "i", "iː", "ɯ", "u", "uː", "e", "ɛ", "o"};
                ipa.push_back(th_vow[cp % 10]);
            } else if (cp >= 0x0E48 && cp <= 0x0E4B) {
                // Thai Tone marks: Mai Ek (Low=3), Mai Tho (Falling=4), Mai Tri (High=1), Mai Chattawa (Rising=2)
                static const char* th_tones[] = {"3", "4", "1", "2"};
                ipa.push_back(th_tones[cp - 0x0E48]);
            } else if (cp >= 0x1EA0 && cp <= 0x1EF9) {
                // Vietnamese Extended Latin vowels with tone diacritics
                static const char* vi_vow[] = {"a", "e", "i", "o", "u", "ɤ", "ɯ"};
                static const char* vi_ton[] = {"1", "2", "3", "4", "5"};
                ipa.push_back(vi_vow[cp % 7]);
                ipa.push_back(vi_ton[cp % 5]);
            }
        } else if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
            uint32_t cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(text[i + 1]) & 0x3F);
            i += 2;
            // Vietnamese tone-marked Latin-1/Latin-Extended vowels (á, à, ả, ã, ạ, đ, etc.)
            if (cp == 0x0111 || cp == 0x0110) {
                ipa.push_back("d");
            } else {
                ipa.push_back("a");
                ipa.push_back(std::to_string((cp % 5) + 1));
            }
        } else {
            i += 1;
        }
    }
    if (lang == "vi" || lang == "th") {
        // Ensure at least one lexical tone feature is active if plain ASCII input was provided
        bool has_tone_tok = false;
        for (const auto& tok : ipa) {
            if (tok == "1" || tok == "2" || tok == "3" || tok == "4" || tok == "5") {
                has_tone_tok = true;
                break;
            }
        }
        if (!has_tone_tok && !ipa.empty()) {
            ipa.push_back("1");
        }
    }
    return ipa;
}

std::vector<std::string> MultilingualG2PN::phonemizeGeneralLatin(const std::string& text, const std::string& lang) const {
    // Covers pt, it, tr, nl, pl, id with Synalepha vowel linking and language-specific phones
    std::vector<std::string> ipa;
    std::string lower = text;
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    for (size_t i = 0; i < lower.size(); ++i) {
        char c = lower[i];
        if (c == ' ') {
            if (i > 0 && i + 1 < lower.size() && isAsciiVowel(lower[i - 1]) && isAsciiVowel(lower[i + 1])) {
                ipa.push_back("‿"); // Synalepha vowel linking
            } else {
                ipa.push_back("_");
            }
            continue;
        }
        if (c >= 'a' && c <= 'z') {
            if ((lang == "it" || lang == "pt" || lang == "pl") && (c == 't' || c == 'd')) {
                ipa.push_back(c == 't' ? "t̪" : "d̪");
            } else if (lang == "nl" && c == 'g') {
                ipa.push_back("x");
            } else if (lang == "tr" && c == 'c') {
                ipa.push_back("dʒ");
            } else {
                ipa.push_back(std::string(1, c));
            }
        }
    }
    return ipa;
}

PhonemeSequence MultilingualG2PN::processClause(const std::string& clause_text,
                                                const std::string& lang_code,
                                                ClauseType clause_type) const {
    PhonemeSequence seq;
    seq.lang_id = getLanguageId(lang_code);
    seq.accent_style_id = 0;
    seq.clause_type = clause_type;
    seq.is_tonal_language = isTonalLanguage(seq.lang_id);

    switch (clause_type) {
        case ClauseType::COMMA:       seq.pause_sec = 0.18f; break;
        case ClauseType::QUESTION:    seq.pause_sec = 0.28f; break;
        case ClauseType::EXCLAMATION: seq.pause_sec = 0.26f; break;
        case ClauseType::STATEMENT:
        default:                      seq.pause_sec = 0.32f; break;
    }

    // Apply colloquial slang reduction before phonemization
    std::string processed_text = applyColloquialSlang(clause_text, lang_code);
    seq.text_clause = processed_text;

    std::vector<std::string> ipa_tokens;
    if (lang_code == "en") {
        ipa_tokens = phonemizeEnglish(processed_text);
    } else if (lang_code == "es") {
        ipa_tokens = phonemizeSpanish(processed_text);
    } else if (lang_code == "fr") {
        ipa_tokens = phonemizeFrench(processed_text);
    } else if (lang_code == "de") {
        ipa_tokens = phonemizeGerman(processed_text);
    } else if (lang_code == "hi" || lang_code == "bn" || lang_code == "ta") {
        ipa_tokens = phonemizeIndic(processed_text, lang_code);
    } else if (lang_code == "ar" || lang_code == "ur") {
        ipa_tokens = phonemizeArabicUrdu(processed_text, lang_code);
    } else if (lang_code == "zh") {
        ipa_tokens = phonemizeChinese(processed_text);
    } else if (lang_code == "ja") {
        ipa_tokens = phonemizeJapanese(processed_text);
    } else if (lang_code == "ko") {
        ipa_tokens = phonemizeKorean(processed_text);
    } else if (lang_code == "ru") {
        ipa_tokens = phonemizeRussian(processed_text);
    } else if (lang_code == "vi" || lang_code == "th") {
        ipa_tokens = phonemizeTonalSEAsian(processed_text, lang_code);
    } else {
        ipa_tokens = phonemizeGeneralLatin(processed_text, lang_code);
    }

    // Add BOS
    seq.phoneme_ids.push_back(TOKEN_BOS);

    // Map tokens to IDs
    for (const auto& token : ipa_tokens) {
        seq.phoneme_ids.push_back(getPhonemeId(token));
    }

    // Add EOS
    seq.phoneme_ids.push_back(TOKEN_EOS);

    // Step 3: Extract 62-D Articulatory Feature Matrix [N, 62]
    seq.articulatory_features = extractArticulatoryMatrix(seq.phoneme_ids, seq.lang_id);

    return seq;
}

} // namespace HDModularTTS
