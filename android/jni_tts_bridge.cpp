#include "HDModularTTSEngine.h"

#include <jni.h>
#include <string>
#include <vector>
#include <algorithm>
#include <android/log.h>

#define TAG "HDModularTTS_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static std::string jstringToUtf8(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* utf = env->GetStringUTFChars(jstr, nullptr);
    std::string str(utf);
    env->ReleaseStringUTFChars(jstr, utf);
    return str;
}

extern "C" {

/**
 * Initializes the HDModularTTSEngine using zero-copy OS mmap.
 * Signature: (Ljava/lang/String;Ljava/lang/String;ZI)J
 */
JNIEXPORT jlong JNICALL
Java_com_antigravity_tts_NativeTTS_nativeInit(JNIEnv* env,
                                              jobject /* thiz */,
                                              jstring jmodels_dir,
                                              jstring jspeaker_lock_path,
                                              jboolean use_vulkan,
                                              jint num_threads) {
    std::string models_dir = jstringToUtf8(env, jmodels_dir);
    std::string speaker_lock = jstringToUtf8(env, jspeaker_lock_path);

    LOGI("Initializing HDModularTTSEngine from: %s (Vulkan=%d, Threads=%d)",
         models_dir.c_str(), (int)use_vulkan, (int)num_threads);

    HDModularTTS::EngineConfig config;
    config.models_dir = models_dir;
    config.use_vulkan = (use_vulkan == JNI_TRUE);
    config.num_threads = static_cast<int>(num_threads);
    config.sample_rate = 24000;
    config.crossfade_samples = 512;

    auto* engine = new HDModularTTS::HDModularTTSEngine(config);
    if (!engine->init()) {
        LOGE("Failed to initialize HDModularTTSEngine via mmap!");
        delete engine;
        return 0;
    }

    LOGI("HDModularTTSEngine initialized successfully.");
    return reinterpret_cast<jlong>(engine);
}

/**
 * Streams synthesized audio chunks directly to a Java callback for AudioTrack feeding.
 * Signature: (JLjava/lang/String;Ljava/lang/String;Lcom/antigravity/tts/NativeTTS$AudioChunkListener;)Z
 */
JNIEXPORT jboolean JNICALL
Java_com_antigravity_tts_NativeTTS_nativeSynthesizeStream(JNIEnv* env,
                                                          jobject /* thiz */,
                                                          jlong engine_ptr,
                                                          jstring jtext,
                                                          jstring jlang_code,
                                                          jobject jcallback) {
    auto* engine = reinterpret_cast<HDModularTTS::HDModularTTSEngine*>(engine_ptr);
    if (!engine || !engine->isReady()) {
        LOGE("Invalid or uninitialized engine pointer in nativeSynthesizeStream");
        return JNI_FALSE;
    }

    std::string text = jstringToUtf8(env, jtext);
    std::string lang = jstringToUtf8(env, jlang_code);

    jclass callback_class = env->GetObjectClass(jcallback);
    jmethodID on_audio_chunk_mid = env->GetMethodID(callback_class, "onAudioChunk", "([BZ)V");
    if (!on_audio_chunk_mid) {
        LOGE("Callback method 'onAudioChunk([BZ)V' not found");
        return JNI_FALSE;
    }

    bool status = engine->synthesizeStream(text, lang, [&](const std::vector<float>& pcm_chunk, bool is_final) {
        if (pcm_chunk.empty()) return;

        // Convert float32 [-1.0, 1.0] to 16-bit PCM byte array (little-endian)
        size_t byte_count = pcm_chunk.size() * sizeof(int16_t);
        jbyteArray byte_array = env->NewByteArray(static_cast<jsize>(byte_count));
        jbyte* raw_bytes = env->GetByteArrayElements(byte_array, nullptr);

        int16_t* pcm16 = reinterpret_cast<int16_t*>(raw_bytes);
        for (size_t i = 0; i < pcm_chunk.size(); ++i) {
            float s = std::max(-1.0f, std::min(1.0f, pcm_chunk[i]));
            pcm16[i] = static_cast<int16_t>(s * 32767.0f);
        }

        env->ReleaseByteArrayElements(byte_array, raw_bytes, 0);

        // Invoke Java callback
        env->CallVoidMethod(jcallback, on_audio_chunk_mid, byte_array, is_final ? JNI_TRUE : JNI_FALSE);
        env->DeleteLocalRef(byte_array);
    });

    return status ? JNI_TRUE : JNI_FALSE;
}

/**
 * One-shot synthesis returning full audio buffer as 16-bit PCM byte array.
 * Signature: (JLjava/lang/String;Ljava/lang/String;)[B
 */
JNIEXPORT jbyteArray JNICALL
Java_com_antigravity_tts_NativeTTS_nativeSynthesizeWav(JNIEnv* env,
                                                       jobject /* thiz */,
                                                       jlong engine_ptr,
                                                       jstring jtext,
                                                       jstring jlang_code) {
    auto* engine = reinterpret_cast<HDModularTTS::HDModularTTSEngine*>(engine_ptr);
    if (!engine) return nullptr;

    std::string text = jstringToUtf8(env, jtext);
    std::string lang = jstringToUtf8(env, jlang_code);

    std::vector<float> pcm_samples;
    if (!engine->synthesize(text, lang, pcm_samples) || pcm_samples.empty()) {
        return nullptr;
    }

    size_t byte_count = pcm_samples.size() * sizeof(int16_t);
    jbyteArray result = env->NewByteArray(static_cast<jsize>(byte_count));
    jbyte* raw_bytes = env->GetByteArrayElements(result, nullptr);

    int16_t* pcm16 = reinterpret_cast<int16_t*>(raw_bytes);
    for (size_t i = 0; i < pcm_samples.size(); ++i) {
        float s = std::max(-1.0f, std::min(1.0f, pcm_samples[i]));
        pcm16[i] = static_cast<int16_t>(s * 32767.0f);
    }

    env->ReleaseByteArrayElements(result, raw_bytes, 0);
    return result;
}

/**
 * Sets the dynamic mood/emotion mode for synthesis.
 * Signature: (JI)V
 */
JNIEXPORT void JNICALL
Java_com_antigravity_tts_NativeTTS_nativeSetMood(JNIEnv* /* env */,
                                                 jobject /* thiz */,
                                                 jlong engine_ptr,
                                                 jint mood_id) {
    auto* engine = reinterpret_cast<HDModularTTS::HDModularTTSEngine*>(engine_ptr);
    if (engine) {
        engine->setMood(static_cast<HDModularTTS::Mood>(mood_id));
        LOGI("Set engine mood to ID: %d", (int)mood_id);
    }
}

/**
 * Releases the native engine and frees memory maps.
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_com_antigravity_tts_NativeTTS_nativeRelease(JNIEnv* /* env */,
                                                 jobject /* thiz */,
                                                 jlong engine_ptr) {
    auto* engine = reinterpret_cast<HDModularTTS::HDModularTTSEngine*>(engine_ptr);
    if (engine) {
        LOGI("Releasing HDModularTTSEngine instance");
        delete engine;
    }
}

} // extern "C"

