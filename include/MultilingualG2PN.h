#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <string_view>

namespace HDModularTTS {

/**
 * Clause intonation and sentence classification
 */
enum class ClauseType {
    STATEMENT,   // Period '.' or Hindi Danda '।' (natural declination)
    QUESTION,    // Question '?' (rising intonation)
    EXCLAMATION, // Exclamation '!' (heightened energy and pitch)
    COMMA        // Comma ',' or semicolon ';' (breathing pause)
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
 * PhonemeSequence represents processed linguistic tokens ready for
 * the Native Acoustic Encoder.
 */
struct PhonemeSequence {
    std::vector<int32_t> phoneme_ids;
    int32_t lang_id{0};
    int32_t accent_style_id{0};
    std::string text_clause;
    std::vector<float> articulatory_features; // Size = N * 62 (IMS-Toucan Articulatory Vectors)
};

/**
 * MultilingualG2PN
 * Production C++20 Multilingual Text Normalizer & Grapheme-to-Phoneme Engine.
 * Supports 21 Languages:
 *   en (English), hi (Hindi), es (Spanish), fr (French), de (German),
 *   zh (Chinese), ja (Japanese), ko (Korean), ru (Russian), ar (Arabic),
 *   pt (Portuguese), it (Italian), tr (Turkish), nl (Dutch), pl (Polish),
 *   id (Indonesian), vi (Vietnamese), th (Thai), ta (Tamil), bn (Bengali),
 *   ur (Urdu).
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
    // Ensures Time-To-First-Audio (TTFA) < 80ms on CPU.
    std::vector<std::string> segmentClauses(const std::string& normalized_text, int max_words_per_clause = 5) const;

    // Splits text with intonation tagging (Question, Exclamation, Statement, Comma) & natural breath timing
    std::vector<ClauseSegment> segmentClausesWithIntonation(const std::string& raw_text, const std::string& lang_code) const;

    // Converts a single text clause to IPA phoneme IDs and language routing
    PhonemeSequence processClause(const std::string& clause_text, const std::string& lang_code) const;

    // Resolves language ISO code (e.g. "en", "hi", "es") to integer lang_id (0-20)
    int32_t getLanguageId(const std::string& lang_code) const;

    // Resolves language name / code from integer lang_id
    std::string getLanguageCode(int32_t lang_id) const;

    // Total phoneme vocabulary size
    size_t getVocabSize() const noexcept { return m_ipa_to_id.size(); }

    // Map single IPA token to integer ID
    int32_t getPhonemeId(const std::string& ipa_symbol) const;

    // Articulatory 62-D Vector Lookup for ToucanTTS
    const float* getArticulatoryVector(int32_t phoneme_id) const;
    std::vector<float> extractArticulatoryMatrix(const std::vector<int32_t>& phoneme_ids) const;

    // Special Token IDs
    static constexpr int32_t TOKEN_PAD = 0;
    static constexpr int32_t TOKEN_BOS = 1;
    static constexpr int32_t TOKEN_EOS = 2;
    static constexpr int32_t TOKEN_PAUSE_SHORT = 3;   // ~100ms pause
    static constexpr int32_t TOKEN_PAUSE_MEDIUM = 4;  // ~200ms pause
    static constexpr int32_t TOKEN_PAUSE_LONG = 5;    // ~350ms pause

private:
    void initVocab();
    void initLanguageMap();
    void initArticulatoryTable();

    // Normalizers
    std::string expandNumbers(const std::string& text, const std::string& lang) const;
    std::string expandCurrencies(const std::string& text, const std::string& lang) const;
    std::string expandContractions(const std::string& text, const std::string& lang) const;
    std::string cleanPunctuation(const std::string& text) const;

    // Language specific phonemizers
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
    std::vector<std::string> phonemizeGeneralLatin(const std::string& text, const std::string& lang) const;

    std::unordered_map<std::string, int32_t> m_ipa_to_id;
    std::unordered_map<std::string, int32_t> m_lang_to_id;
    std::vector<std::string> m_id_to_lang;
    std::vector<std::vector<float>> m_articulatory_table; // [512, 62] table
};

} // namespace HDModularTTS

