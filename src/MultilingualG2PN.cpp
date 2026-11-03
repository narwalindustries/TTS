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
    // 21 supported languages
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

void MultilingualG2PN::initVocab() {
    // Special Tokens
    m_ipa_to_id["<pad>"] = TOKEN_PAD;
    m_ipa_to_id["<bos>"] = TOKEN_BOS;
    m_ipa_to_id["<eos>"] = TOKEN_EOS;
    m_ipa_to_id["_"]     = TOKEN_PAUSE_SHORT;
    m_ipa_to_id["__"]    = TOKEN_PAUSE_MEDIUM;
    m_ipa_to_id["___"]   = TOKEN_PAUSE_LONG;

    // Universal IPA Phoneme Set (Vowels, Consonants, Tones, Accents)
    const std::vector<std::string> ipa_symbols = {
        // Vowels
        "a", "e", "i", "o", "u", "y", "ə", "ɛ", "ɪ", "ʊ", "ʌ", "ɔ", "æ", "ɑ",
        "ɜ", "ɨ", "ʉ", "ɯ", "ʏ", "ø", "œ", "ɐ", "ɤ", "ã", "ẽ", "ĩ", "õ", "ũ",
        // Consonants
        "p", "b", "t", "d", "k", "g", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ",
        "h", "m", "n", "ŋ", "l", "r", "j", "w", "tʃ", "dʒ", "ts", "dz",
        // Indic Retroflex & Aspirated Consonants
        "ʈ", "ɖ", "ɳ", "ɭ", "ɽ", "ʂ", "ʐ", "c", "ɟ", "ɲ", "ʔ", "x", "ɣ", "q",
        "χ", "ʁ", "ʕ", "ħ", "ɬ", "ɮ", "bʱ", "dʱ", "ɖʱ", "gʱ", "dʒʱ", "pʰ", "tʰ",
        "ʈʰ", "kʰ", "tʃʰ",
        // Tones and Stress Accents
        "ˈ", "ˌ", "ː", "˥", "˦", "˧", "˨", "˩", "1", "2", "3", "4", "5",
        // Nasal and Length Markers
        "̃", "̆", "̥", "̬", "ʰ", "ʲ", "ʷ", "ˤ"
    };

    int32_t id = 6;
    for (const auto& sym : ipa_symbols) {
        if (m_ipa_to_id.find(sym) == m_ipa_to_id.end()) {
            m_ipa_to_id[sym] = id++;
        }
    }

    // Fill remaining up to 512 for ASCII characters & extensions
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
    // Fallback: character-by-character lookup
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
    // 62-Dimensional Articulatory Phoneme Feature Specification (IMS-Toucan / PanPhon)
    // Dims 0..21: Place of Articulation (bilabial, labiodental, dental, alveolar, postalveolar, retroflex, etc.)
    // Dims 22..35: Manner (stop, nasal, fricative, affricate, lateral, trill, tap/flap)
    // Dims 36..40: Voicing & Aspiration
    // Dims 41..54: Vowel Features (syllabic, high, low, front, back, rounded)
    // Dims 55..61: Tone Features (high, mid, low, rising, falling, contour, neutral)
    m_articulatory_table.assign(512, std::vector<float>(ARTICULATORY_DIM, 0.0f));

    for (const auto& [sym, id] : m_ipa_to_id) {
        if (id < 0 || id >= 512) continue;
        auto& feat = m_articulatory_table[id];

        // Vowels
        if (sym == "a" || sym == "æ" || sym == "ɑ" || sym == "ɐ") {
            feat[0] = 1.0f; feat[22] = 1.0f; // Syllabic, Low vowel
            if (sym == "æ" || sym == "a") feat[23] = 1.0f; // Front
            else feat[24] = 1.0f; // Back
        } else if (sym == "e" || sym == "ɛ" || sym == "ə" || sym == "ø" || sym == "œ") {
            feat[0] = 1.0f; feat[23] = 1.0f; // Syllabic, Mid/Front
            if (sym == "ø" || sym == "œ") feat[25] = 1.0f; // Rounded
        } else if (sym == "i" || sym == "ɪ" || sym == "y" || sym == "ɨ") {
            feat[0] = 1.0f; feat[21] = 1.0f; feat[23] = 1.0f; // Syllabic, High, Front
            if (sym == "y") feat[25] = 1.0f; // Rounded
        } else if (sym == "o" || sym == "ɔ" || sym == "u" || sym == "ʊ" || sym == "ɯ") {
            feat[0] = 1.0f; feat[24] = 1.0f; feat[25] = 1.0f; // Syllabic, Back, Round
            if (sym == "u" || sym == "ʊ") feat[21] = 1.0f; // High
        }
        // Consonants - Dental (त, थ, द, ध, θ, ð) - HARD-LOCKED
        else if (sym == "t̪" || sym == "d̪" || sym == "θ" || sym == "ð") {
            feat[1] = 1.0f; feat[10] = 1.0f; feat[11] = 1.0f; // Consonant, Coronal, Dental!
            if (sym == "t̪" || sym == "d̪") feat[4] = 1.0f; // Stop
            if (sym == "θ" || sym == "ð") feat[3] = 1.0f; // Fricative
            if (sym == "d̪" || sym == "ð") feat[27] = 1.0f; // Voiced
        }
        // Consonants - Retroflex (ट, ठ, ड, ढ, ʈ, ɖ, ɳ, ɭ, ɽ, ʂ, ʐ, ɻ) - HARD-LOCKED
        else if (sym == "ʈ" || sym == "ɖ" || sym == "ɳ" || sym == "ɭ" || sym == "ɽ" || sym == "ʂ" || sym == "ʐ" || sym == "ɻ") {
            feat[1] = 1.0f; feat[10] = 1.0f; feat[14] = 1.0f; // Consonant, Coronal, Retroflex!
            if (sym == "ʈ" || sym == "ɖ") feat[4] = 1.0f;
            if (sym == "ɳ") feat[5] = 1.0f; // Nasal
            if (sym == "ɭ" || sym == "ɻ") feat[6] = 1.0f; // Lateral/Approximant
            if (sym == "ɽ") feat[29] = 1.0f; // Tap/flap
            if (sym == "ɖ" || sym == "ɳ" || sym == "ɭ" || sym == "ʐ" || sym == "ɻ") feat[27] = 1.0f; // Voiced
        }
        // Consonants - Uvular & Pharyngeal (Urdu/Arabic nuqtas: क़, ख़, ग़, q, χ, ʁ, ħ, ʕ)
        else if (sym == "q" || sym == "χ" || sym == "ʁ") {
            feat[1] = 1.0f; feat[15] = 1.0f; feat[18] = 1.0f; // Uvular
            if (sym == "q") feat[4] = 1.0f;
            if (sym == "χ" || sym == "ʁ") feat[3] = 1.0f;
            if (sym == "ʁ") feat[27] = 1.0f;
        } else if (sym == "ħ" || sym == "ʕ") {
            feat[1] = 1.0f; feat[19] = 1.0f; feat[3] = 1.0f; // Pharyngeal Fricative
            if (sym == "ʕ") feat[27] = 1.0f;
        }
        // Consonants - Alveolar (English t, d, s, z, n, l, r)
        else if (sym == "t" || sym == "d" || sym == "s" || sym == "z" || sym == "n" || sym == "l" || sym == "r" || sym == "ɾ") {
            feat[1] = 1.0f; feat[10] = 1.0f; feat[12] = 1.0f; // Coronal, Alveolar
            if (sym == "t" || sym == "d") feat[4] = 1.0f;
            if (sym == "s" || sym == "z") feat[3] = 1.0f;
            if (sym == "n") feat[5] = 1.0f;
            if (sym == "l") feat[6] = 1.0f;
            if (sym == "ɾ") feat[29] = 1.0f; // Flap/tap (English flapped-T)
            if (sym == "d" || sym == "z" || sym == "n" || sym == "l" || sym == "r" || sym == "ɾ") feat[27] = 1.0f; // Voiced
        }
        // Consonants - Bilabial & Labiodental (p, b, m, f, v, w)
        else if (sym == "p" || sym == "b" || sym == "m" || sym == "f" || sym == "v" || sym == "w") {
            feat[1] = 1.0f; feat[7] = 1.0f; // Labial
            if (sym == "p" || sym == "b" || sym == "m") feat[8] = 1.0f; // Bilabial
            if (sym == "f" || sym == "v") feat[9] = 1.0f; // Labiodental
            if (sym == "m") feat[5] = 1.0f; // Nasal
            if (sym == "b" || sym == "m" || sym == "v" || sym == "w") feat[27] = 1.0f; // Voiced
        }
        // Consonants - Velar (k, g, ŋ, x, ɣ)
        else if (sym == "k" || sym == "g" || sym == "ŋ" || sym == "x" || sym == "ɣ") {
            feat[1] = 1.0f; feat[15] = 1.0f; feat[17] = 1.0f; // Dorsal, Velar
            if (sym == "k" || sym == "g") feat[4] = 1.0f;
            if (sym == "ŋ") feat[5] = 1.0f;
            if (sym == "g" || sym == "ŋ" || sym == "ɣ") feat[27] = 1.0f; // Voiced
        }

        // Aspirates
        if (sym.find("ʰ") != std::string::npos || sym.find("ʱ") != std::string::npos) {
            feat[28] = 1.0f; // Aspirated
        }

        // Tonal Features (dims 55..61)
        if (sym == "1" || sym == "˥") { feat[55] = 1.0f; } // High level
        else if (sym == "2" || sym == "˦") { feat[56] = 1.0f; } // High rising
        else if (sym == "3" || sym == "˧") { feat[57] = 1.0f; } // Mid / falling-rising
        else if (sym == "4" || sym == "˨") { feat[58] = 1.0f; } // Falling
        else if (sym == "5" || sym == "˩") { feat[59] = 1.0f; } // Neutral / low
    }
}

const float* MultilingualG2PN::getArticulatoryVector(int32_t phoneme_id) const {
    static const float s_zero[ARTICULATORY_DIM] = {0.0f};
    if (phoneme_id >= 0 && phoneme_id < static_cast<int32_t>(m_articulatory_table.size())) {
        return m_articulatory_table[phoneme_id].data();
    }
    return s_zero;
}

std::vector<float> MultilingualG2PN::extractArticulatoryMatrix(const std::vector<int32_t>& phoneme_ids) const {
    std::vector<float> mat;
    mat.reserve(phoneme_ids.size() * ARTICULATORY_DIM);
    for (int32_t pid : phoneme_ids) {
        const float* vec = getArticulatoryVector(pid);
        mat.insert(mat.end(), vec, vec + ARTICULATORY_DIM);
    }
    return mat;
}

// ============================================================================
// Text Normalization Logic
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
        static const std::vector<std::pair<std::regex, std::string>> en_contractions = {
            {std::regex(R"(\bwon't\b)", std::regex::icase), "will not"},
            {std::regex(R"(\bcan't\b)", std::regex::icase), "cannot"},
            {std::regex(R"(\bn't\b)", std::regex::icase), " not"},
            {std::regex(R"(\b're\b)", std::regex::icase), " are"},
            {std::regex(R"(\b's\b)", std::regex::icase), " is"},
            {std::regex(R"(\b'd\b)", std::regex::icase), " would"},
            {std::regex(R"(\b'll\b)", std::regex::icase), " will"},
            {std::regex(R"(\b've\b)", std::regex::icase), " have"},
            {std::regex(R"(\b'm\b)", std::regex::icase), " am"},
            {std::regex(R"(\bgonna\b)", std::regex::icase), "going to"},
            {std::regex(R"(\bwanna\b)", std::regex::icase), "want to"},
            {std::regex(R"(\bgotta\b)", std::regex::icase), "got to"},
            {std::regex(R"(\bkinda\b)", std::regex::icase), "kind of"},
            {std::regex(R"(\blemme\b)", std::regex::icase), "let me"},
            {std::regex(R"(\bgimme\b)", std::regex::icase), "give me"}
        };
        for (const auto& [pattern, rep] : en_contractions) {
            res = std::regex_replace(res, pattern, rep);
        }
    } else if (lang == "es") {
        res = std::regex_replace(res, std::regex(R"(\bdel\b)", std::regex::icase), "de el");
        res = std::regex_replace(res, std::regex(R"(\bal\b)", std::regex::icase), "a el");
    } else if (lang == "hi") {
        // Natural spoken Hindi phonetic & pronoun normalization
        // Replaces bookish orthography with natural colloquial pronunciations
        res = std::regex_replace(res, std::regex(R"((^|\s)यह(\s|[।,!?]|$))"), "$1ये$2");
        res = std::regex_replace(res, std::regex(R"((^|\s)वह(\s|[।,!?]|$))"), "$1वो$2");
        res = std::regex_replace(res, std::regex(R"(\bलॉक\b)"), "सेट");
        res = std::regex_replace(res, std::regex(R"(\bलहज़ा\b)"), "अंदाज़");
        res = std::regex_replace(res, std::regex(R"( और )"), ", और ");
        res = std::regex_replace(res, std::regex(R"( लेकिन )"), ", लेकिन ");
        res = std::regex_replace(res, std::regex(R"( कि )"), ", कि ");
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
    // Map major punctuation (including Hindi Purna Viram '।') to long pauses
    res = std::regex_replace(res, std::regex(R"([\.\!\?।]+)"), " ___ ");
    res = std::regex_replace(res, std::regex(R"([\,\;\:\—\-]+)"), " _ ");
    // Remove unpronounceable symbols
    res = std::regex_replace(res, std::regex(R"([\"\'\(\)\[\]\{\}\<\>\*\#\@\^\~\`])"), " ");
    return res;
}

std::string MultilingualG2PN::applyColloquialSlang(const std::string& text, const std::string& lang) const {
    std::string res = text;
    if (lang == "hi") {
        // Colloquial speech contractions (Schwa deletion and spoken slang)
        res = std::regex_replace(res, std::regex(R"(क्या\s+कर\s+रहा\s+है)"), "क्या कर रा है");
        res = std::regex_replace(res, std::regex(R"(क्या\s+कर\s+रही\s+है)"), "क्या कर री है");
        res = std::regex_replace(res, std::regex(R"(समझ\s+नहीं\s+आया)"), "समझ नई आया");
        res = std::regex_replace(res, std::regex(R"(कह\s+रहा\s+था)"), "केह रा था");
        res = std::regex_replace(res, std::regex(R"(कह\s+रही\s+थी)"), "केह री थी");
        res = std::regex_replace(res, std::regex(R"(\bनहीं\b)"), "नई");
        res = std::regex_replace(res, std::regex(R"(\bबहुत\b)"), "बोत");
    } else if (lang == "ur") {
        // Urdu colloquial mappings
        res = std::regex_replace(res, std::regex(R"(کیا\s+کر\s+رہا\s+ہے)"), "کیا کر را ہے");
        res = std::regex_replace(res, std::regex(R"(سمجھ\s+نہیں\s+آیا)"), "سمجھ نئی آیا");
    } else if (lang == "ta") {
        // Tamil colloquial tanglish reductions (e.g. final nasalization)
        res = std::regex_replace(res, std::regex(R"(வந்தேன்)"), "வந்தேఁ");
        res = std::regex_replace(res, std::regex(R"(போனேன்)"), "போனேఁ");
    } else if (lang == "en") {
        res = std::regex_replace(res, std::regex(R"(\bgoing to\b)", std::regex::icase), "gonna");
        res = std::regex_replace(res, std::regex(R"(\bwant to\b)", std::regex::icase), "wanna");
        res = std::regex_replace(res, std::regex(R"(\bgot to\b)", std::regex::icase), "gotta");
        res = std::regex_replace(res, std::regex(R"(\bkind of\b)", std::regex::icase), "kinda");
        res = std::regex_replace(res, std::regex(R"(\blet me\b)", std::regex::icase), "lemme");
        res = std::regex_replace(res, std::regex(R"(\bgive me\b)", std::regex::icase), "gimme");
        // Flapped-T rule in common words
        res = std::regex_replace(res, std::regex(R"(\bwater\b)", std::regex::icase), "wader");
        res = std::regex_replace(res, std::regex(R"(\bbutter\b)", std::regex::icase), "budder");
        res = std::regex_replace(res, std::regex(R"(\bbetter\b)", std::regex::icase), "bedder");
        res = std::regex_replace(res, std::regex(R"(\bcity\b)", std::regex::icase), "cidy");
    } else if (lang == "fr") {
        res = std::regex_replace(res, std::regex(R"(\bJe ne sais pas\b)", std::regex::icase), "J'sais pas");
        res = std::regex_replace(res, std::regex(R"(\bIl n'y a pas\b)", std::regex::icase), "Y'a pas");
    } else if (lang == "es") {
        res = std::regex_replace(res, std::regex(R"(\bpara\b)", std::regex::icase), "pa'");
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

    // First normalize text (currencies, numbers, colloquial forms)
    std::string text = raw_text;
    text = expandCurrencies(text, lang_code);
    text = expandContractions(text, lang_code);
    text = expandNumbers(text, lang_code);

    // Delimiters regex capturing intonation markers: [.,;:!?|।]
    std::regex re_delim(R"(([\,\;\:\.\!\?।\n]+))");
    std::sregex_token_iterator it(text.begin(), text.end(), re_delim, {-1, 0});
    std::sregex_token_iterator end;

    std::string cur_text;
    for (; it != end; ++it) {
        std::string token = *it;
        if (token.empty()) continue;

        // Check if this token is a delimiter
        bool is_delim = false;
        for (char c : token) {
            if (c == ',' || c == ';' || c == ':' || c == '.' || c == '!' || c == '?' || c == '\n') {
                is_delim = true;
                break;
            }
        }
        // Also check Hindi Purna Viram (3-byte UTF-8 for '।': 0xE0 0xA5 0xA4)
        if (token.find("\xE0\xA5\xA4") != std::string::npos) {
            is_delim = true;
        }

        if (!is_delim) {
            cur_text += token;
        } else {
            // Determine ClauseType from delimiter
            ClauseType ctype = ClauseType::STATEMENT;
            float pause_sec = 0.20f;
            bool has_breath = false;

            if (token.find('?') != std::string::npos) {
                ctype = ClauseType::QUESTION;
                pause_sec = 0.24f;
                has_breath = true;
            } else if (token.find('!') != std::string::npos) {
                ctype = ClauseType::EXCLAMATION;
                pause_sec = 0.22f;
                has_breath = true;
            } else if (token.find('.') != std::string::npos || token.find("\xE0\xA5\xA4") != std::string::npos || token.find('\n') != std::string::npos) {
                ctype = ClauseType::STATEMENT;
                pause_sec = 0.22f;
                has_breath = true;
            } else if (token.find(',') != std::string::npos || token.find(';') != std::string::npos || token.find(':') != std::string::npos) {
                ctype = ClauseType::COMMA;
                pause_sec = 0.12f;
                has_breath = false;
            }

            // Trim cur_text
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

    // Trailing text without punctuation
    size_t start = cur_text.find_first_not_of(" \t\r\n");
    size_t finish = cur_text.find_last_not_of(" \t\r\n");
    if (start != std::string::npos && finish >= start) {
        std::string clean = cur_text.substr(start, finish - start + 1);
        if (!clean.empty()) {
            result.push_back({clean, ClauseType::STATEMENT, 0.20f, false});
        }
    }

    return result;
}

// ============================================================================
// Multilingual IPA Phonemizers
// ============================================================================

std::vector<std::string> MultilingualG2PN::phonemizeEnglish(const std::string& text) const {
    std::vector<std::string> ipa;
    for (char c : text) {
        c = std::tolower(static_cast<unsigned char>(c));
        if (c >= 'a' && c <= 'z') {
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
        } else if (c == ' ') {
            ipa.push_back("_");
        }
    }
    return ipa;
}

std::vector<std::string> MultilingualG2PN::phonemizeSpanish(const std::string& text) const {
    std::vector<std::string> ipa;
    for (char c : text) {
        c = std::tolower(static_cast<unsigned char>(c));
        if (c >= 'a' && c <= 'z') {
            switch (c) {
                case 'a': ipa.push_back("a"); break;
                case 'e': ipa.push_back("e"); break;
                case 'i': ipa.push_back("i"); break;
                case 'o': ipa.push_back("o"); break;
                case 'u': ipa.push_back("u"); break;
                case 'j': ipa.push_back("x"); break;
                case 'v': ipa.push_back("b"); break;
                case 'z': ipa.push_back("θ"); break;
                default: ipa.push_back(std::string(1, c)); break;
            }
        } else if (c == ' ') {
            ipa.push_back("_");
        }
    }
    return ipa;
}

std::vector<std::string> MultilingualG2PN::phonemizeFrench(const std::string& text) const {
    std::vector<std::string> ipa;
    for (char c : text) {
        c = std::tolower(static_cast<unsigned char>(c));
        if (c >= 'a' && c <= 'z') {
            switch (c) {
                case 'u': ipa.push_back("y"); break;
                case 'r': ipa.push_back("ʁ"); break;
                case 'j': ipa.push_back("ʒ"); break;
                default: ipa.push_back(std::string(1, c)); break;
            }
        } else if (c == ' ') {
            ipa.push_back("_");
        }
    }
    return ipa;
}

std::vector<std::string> MultilingualG2PN::phonemizeGerman(const std::string& text) const {
    std::vector<std::string> ipa;
    for (char c : text) {
        c = std::tolower(static_cast<unsigned char>(c));
        if (c >= 'a' && c <= 'z') {
            switch (c) {
                case 'w': ipa.push_back("v"); break;
                case 'v': ipa.push_back("f"); break;
                case 'z': ipa.push_back("ts"); break;
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
    // Map Indic Unicode code points to IPA phonemes
    // Covers Devanagari (Hindi 0x0900-0x097F), Bengali (0x0980-0x09FF), Tamil (0x0B80-0x0BFF)
    for (size_t i = 0; i < text.size();) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 128) {
            if (c == ' ') ipa.push_back("_");
            else if (std::isalnum(c)) ipa.push_back(std::string(1, std::tolower(c)));
            i += 1;
        } else if ((c & 0xE0) == 0xE0 && i + 2 < text.size()) {
            // 3-byte UTF-8 character (Indic script block)
            uint32_t codepoint = ((c & 0x0F) << 12) |
                                 ((static_cast<unsigned char>(text[i+1]) & 0x3F) << 6) |
                                 (static_cast<unsigned char>(text[i+2]) & 0x3F);
            i += 3;

            // Devanagari (Hindi)
            if (codepoint >= 0x0905 && codepoint <= 0x0914) {
                // Vowels
                static const char* dev_vowels[] = {"a", "ɑː", "i", "iː", "u", "uː", "ri", "e", "eː", "ai", "o", "oː", "au"};
                size_t idx = codepoint - 0x0905;
                if (idx < 13) ipa.push_back(dev_vowels[idx]);
            } else if (codepoint >= 0x0915 && codepoint <= 0x0939) {
                // Consonants (inherent schwa ə)
                static const char* dev_cons[] = {
                    "k", "kʰ", "g", "gʱ", "ŋ",
                    "tʃ", "tʃʰ", "dʒ", "dʒʱ", "ɲ",
                    "ʈ", "ʈʰ", "ɖ", "ɖʱ", "ɳ",
                    "t", "tʰ", "d", "dʱ", "n",
                    "n", "p", "pʰ", "b", "bʱ", "m",
                    "j", "r", "l", "ɭ", "v", "ʃ", "ʂ", "s", "h"
                };
                size_t idx = codepoint - 0x0915;
                if (idx < 35) {
                    ipa.push_back(dev_cons[idx]);
                    ipa.push_back("ə");
                }
            } else if (codepoint >= 0x093E && codepoint <= 0x094C) {
                // Matras (replace inherent schwa)
                if (!ipa.empty() && ipa.back() == "ə") ipa.pop_back();
                static const char* matras[] = {"ɑː", "i", "iː", "u", "uː", "ri", "e", "eː", "ai", "o", "oː", "au"};
                size_t idx = codepoint - 0x093E;
                if (idx < 12) ipa.push_back(matras[idx]);
            } else if (codepoint == 0x094D) {
                // Virama / Halant: delete inherent vowel
                if (!ipa.empty() && ipa.back() == "ə") ipa.pop_back();
            } else if (codepoint == 0x0902) {
                // Anusvara
                ipa.push_back("̃");
            } else {
                ipa.push_back("a");
            }
        } else {
            i += 1;
        }
    }
    return ipa;
}

std::vector<std::string> MultilingualG2PN::phonemizeArabicUrdu(const std::string& text, const std::string& lang) const {
    std::vector<std::string> ipa;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 128) {
            if (c == ' ') ipa.push_back("_");
            i += 1;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
            uint32_t cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(text[i+1]) & 0x3F);
            i += 2;
            switch (cp) {
                case 0x0627: ipa.push_back("aː"); break; // Alif
                case 0x0628: ipa.push_back("b"); break;  // Ba
                case 0x062A: ipa.push_back("t"); break;  // Ta
                case 0x062B: ipa.push_back("θ"); break;  // Tha
                case 0x062C: ipa.push_back("dʒ"); break; // Jim
                case 0x062D: ipa.push_back("ħ"); break;  // Hha
                case 0x062E: ipa.push_back("x"); break;  // Kha
                case 0x062F: ipa.push_back("d"); break;  // Dal
                case 0x0630: ipa.push_back("ð"); break;  // Dhal
                case 0x0631: ipa.push_back("r"); break;  // Ra
                case 0x0632: ipa.push_back("z"); break;  // Zay
                case 0x0633: ipa.push_back("s"); break;  // Sin
                case 0x0634: ipa.push_back("ʃ"); break;  // Shin
                case 0x0635: ipa.push_back("sˤ"); break; // Sad
                case 0x0636: ipa.push_back("dˤ"); break; // Dad
                case 0x0637: ipa.push_back("tˤ"); break; // Taa
                case 0x0638: ipa.push_back("ðˤ"); break; // Zaa
                case 0x0639: ipa.push_back("ʕ"); break;  // Ayn
                case 0x063A: ipa.push_back("ɣ"); break;  // Ghayn
                case 0x0641: ipa.push_back("f"); break;  // Fa
                case 0x0642: ipa.push_back("q"); break;  // Qaf
                case 0x0643: ipa.push_back("k"); break;  // Kaf
                case 0x0644: ipa.push_back("l"); break;  // Lam
                case 0x0645: ipa.push_back("m"); break;  // Mim
                case 0x0646: ipa.push_back("n"); break;  // Nun
                case 0x0647: ipa.push_back("h"); break;  // Ha
                case 0x0648: ipa.push_back("w"); break;  // Waw
                case 0x064A: ipa.push_back("j"); break;  // Ya
                default: break;
            }
        } else {
            i += 1;
        }
    }
    return ipa;
}

std::vector<std::string> MultilingualG2PN::phonemizeChinese(const std::string& text) const {
    // Pinyin & tone representation
    std::vector<std::string> ipa;
    for (char c : text) {
        if (c >= '0' && c <= '9') {
            ipa.push_back(std::string(1, c)); // Tone marker
        } else if (c == ' ') {
            ipa.push_back("_");
        } else if (std::isalpha(static_cast<unsigned char>(c))) {
            ipa.push_back(std::string(1, std::tolower(c)));
        }
    }
    return ipa;
}

std::vector<std::string> MultilingualG2PN::phonemizeJapanese(const std::string& text) const {
    std::vector<std::string> ipa;
    for (char c : text) {
        if (c == ' ') ipa.push_back("_");
        else if (std::isalnum(static_cast<unsigned char>(c))) {
            ipa.push_back(std::string(1, std::tolower(c)));
        }
    }
    return ipa;
}

std::vector<std::string> MultilingualG2PN::phonemizeKorean(const std::string& text) const {
    std::vector<std::string> ipa;
    for (char c : text) {
        if (c == ' ') ipa.push_back("_");
        else if (std::isalnum(static_cast<unsigned char>(c))) {
            ipa.push_back(std::string(1, std::tolower(c)));
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
            i += 1;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
            uint32_t cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(text[i+1]) & 0x3F);
            i += 2;
            // Cyrillic mapping
            switch (cp) {
                case 0x0430: case 0x0410: ipa.push_back("a"); break;
                case 0x0431: case 0x0411: ipa.push_back("b"); break;
                case 0x0432: case 0x0412: ipa.push_back("v"); break;
                case 0x0433: case 0x0413: ipa.push_back("g"); break;
                case 0x0434: case 0x0414: ipa.push_back("d"); break;
                case 0x0435: case 0x0415: ipa.push_back("j"); ipa.push_back("e"); break;
                case 0x0436: case 0x0416: ipa.push_back("ʒ"); break;
                case 0x0437: case 0x0417: ipa.push_back("z"); break;
                case 0x0438: case 0x0418: ipa.push_back("i"); break;
                case 0x043A: case 0x041A: ipa.push_back("k"); break;
                case 0x043B: case 0x041B: ipa.push_back("l"); break;
                case 0x043C: case 0x041C: ipa.push_back("m"); break;
                case 0x043D: case 0x041D: ipa.push_back("n"); break;
                case 0x043E: case 0x041E: ipa.push_back("o"); break;
                case 0x043F: case 0x041F: ipa.push_back("p"); break;
                case 0x0440: case 0x0420: ipa.push_back("r"); break;
                case 0x0441: case 0x0421: ipa.push_back("s"); break;
                case 0x0442: case 0x0422: ipa.push_back("t"); break;
                case 0x0443: case 0x0423: ipa.push_back("u"); break;
                case 0x0444: case 0x0424: ipa.push_back("f"); break;
                case 0x0445: case 0x0425: ipa.push_back("x"); break;
                case 0x0446: case 0x0426: ipa.push_back("ts"); break;
                case 0x0447: case 0x0427: ipa.push_back("tʃ"); break;
                case 0x0448: case 0x0428: ipa.push_back("ʃ"); break;
                default: break;
            }
        } else {
            i += 1;
        }
    }
    return ipa;
}

std::vector<std::string> MultilingualG2PN::phonemizeGeneralLatin(const std::string& text, const std::string& lang) const {
    std::vector<std::string> ipa;
    for (char c : text) {
        c = std::tolower(static_cast<unsigned char>(c));
        if (c >= 'a' && c <= 'z') {
            ipa.push_back(std::string(1, c));
        } else if (c == ' ') {
            ipa.push_back("_");
        }
    }
    return ipa;
}

PhonemeSequence MultilingualG2PN::processClause(const std::string& clause_text, const std::string& lang_code) const {
    PhonemeSequence seq;
    seq.lang_id = getLanguageId(lang_code);
    seq.accent_style_id = 0;
    seq.text_clause = clause_text;

    std::vector<std::string> ipa_tokens;
    if (lang_code == "en") {
        ipa_tokens = phonemizeEnglish(clause_text);
    } else if (lang_code == "es") {
        ipa_tokens = phonemizeSpanish(clause_text);
    } else if (lang_code == "fr") {
        ipa_tokens = phonemizeFrench(clause_text);
    } else if (lang_code == "de") {
        ipa_tokens = phonemizeGerman(clause_text);
    } else if (lang_code == "hi" || lang_code == "bn" || lang_code == "ta") {
        ipa_tokens = phonemizeIndic(clause_text, lang_code);
    } else if (lang_code == "ar" || lang_code == "ur") {
        ipa_tokens = phonemizeArabicUrdu(clause_text, lang_code);
    } else if (lang_code == "zh") {
        ipa_tokens = phonemizeChinese(clause_text);
    } else if (lang_code == "ja") {
        ipa_tokens = phonemizeJapanese(clause_text);
    } else if (lang_code == "ko") {
        ipa_tokens = phonemizeKorean(clause_text);
    } else if (lang_code == "ru") {
        ipa_tokens = phonemizeRussian(clause_text);
    } else {
        ipa_tokens = phonemizeGeneralLatin(clause_text, lang_code);
    }

    // Add BOS
    seq.phoneme_ids.push_back(TOKEN_BOS);

    // Map tokens to IDs
    for (const auto& token : ipa_tokens) {
        seq.phoneme_ids.push_back(getPhonemeId(token));
    }

    // Add EOS
    seq.phoneme_ids.push_back(TOKEN_EOS);

    // Extract 62-D Articulatory Feature Matrix (IMS-Toucan)
    seq.articulatory_features = extractArticulatoryMatrix(seq.phoneme_ids);

    return seq;
}

} // namespace HDModularTTS

