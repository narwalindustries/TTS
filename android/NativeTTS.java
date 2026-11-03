package com.antigravity.tts;

import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.os.Build;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * NativeTTS
 * Production Android wrapper for HD Modular Multilingual TTS engine.
 * Connects to Tencent NCNN modular backend via JNI and streams PCM audio
 * directly into low-latency AudioTrack for sub-80ms playback.
 */
public class NativeTTS {
    private static final String TAG = "NativeTTS";
    private static final int SAMPLE_RATE = 24000;

    static {
        try {
            System.loadLibrary("modular_tts_jni");
            Log.i(TAG, "Loaded libmodular_tts_jni.so successfully.");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Could not load libmodular_tts_jni: " + e.getMessage());
        }
    }

    public interface AudioChunkListener {
        void onAudioChunk(byte[] pcmData, boolean isFinal);
    }

    private long mEngineHandle = 0;
    private AudioTrack mAudioTrack;
    private boolean mIsStreaming = false;

    public NativeTTS() {}

    /**
     * Initialize engine with models directory using default native pretrained female voices.
     */
    public boolean init(String modelsDir, boolean useVulkan, int numThreads) {
        return init(modelsDir, "", useVulkan, numThreads);
    }

    /**
     * Initialize engine with models directory and optional parameters.
     */
    public boolean init(String modelsDir, String speakerLockPath, boolean useVulkan, int numThreads) {
        if (mEngineHandle != 0) {
            release();
        }
        mEngineHandle = nativeInit(modelsDir, speakerLockPath, useVulkan, numThreads);
        if (mEngineHandle != 0) {
            initAudioTrack();
            return true;
        }
        return false;
    }

    private void initAudioTrack() {
        int minBufferSize = AudioTrack.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT
        );

        int bufferSize = Math.max(minBufferSize, SAMPLE_RATE / 4); // ~250ms buffer

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            mAudioTrack = new AudioTrack.Builder()
                    .setAudioAttributes(new AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build())
                    .setAudioFormat(new AudioFormat.Builder()
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setSampleRate(SAMPLE_RATE)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .build())
                    .setBufferSizeInBytes(bufferSize)
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
                    .build();
        } else {
            mAudioTrack = new AudioTrack(
                    AudioManager.STREAM_MUSIC,
                    SAMPLE_RATE,
                    AudioFormat.CHANNEL_OUT_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    bufferSize,
                    AudioTrack.MODE_STREAM
            );
        }
    }

    /**
     * Synthesize and immediately play audio in streaming mode (<80ms TTFA).
     */
    public boolean speak(String text, String langCode) {
        if (mEngineHandle == 0 || mAudioTrack == null) {
            Log.e(TAG, "Engine not initialized");
            return false;
        }

        mAudioTrack.play();
        mIsStreaming = true;

        boolean success = nativeSynthesizeStream(mEngineHandle, text, langCode, new AudioChunkListener() {
            @Override
            public void onAudioChunk(byte[] pcmData, boolean isFinal) {
                if (mAudioTrack != null && mIsStreaming) {
                    mAudioTrack.write(pcmData, 0, pcmData.length, AudioTrack.WRITE_NON_BLOCKING);
                }
                if (isFinal) {
                    mIsStreaming = false;
                }
            }
        });

        return success;
    }

    /**
     * Synthesize and save directly to a WAV file on device storage.
     */
    public boolean synthesizeToFile(String text, String langCode, File outputFile) {
        if (mEngineHandle == 0) return false;

        byte[] pcmData = nativeSynthesizeWav(mEngineHandle, text, langCode);
        if (pcmData == null || pcmData.length == 0) return false;

        try (FileOutputStream fos = new FileOutputStream(outputFile)) {
            writeWavHeader(fos, pcmData.length, SAMPLE_RATE, 1, 16);
            fos.write(pcmData);
            return true;
        } catch (IOException e) {
            Log.e(TAG, "Failed to write WAV file: " + e.getMessage());
            return false;
        }
    }

    private void writeWavHeader(FileOutputStream out, int audioLength, int sampleRate, int channels, int bitsPerSample) throws IOException {
        int byteRate = sampleRate * channels * (bitsPerSample / 8);
        int totalDataLen = audioLength + 36;

        byte[] header = new byte[44];
        header[0] = 'R'; header[1] = 'I'; header[2] = 'F'; header[3] = 'F';
        header[4] = (byte) (totalDataLen & 0xff);
        header[5] = (byte) ((totalDataLen >> 8) & 0xff);
        header[6] = (byte) ((totalDataLen >> 16) & 0xff);
        header[7] = (byte) ((totalDataLen >> 24) & 0xff);
        header[8] = 'W'; header[9] = 'A'; header[10] = 'V'; header[11] = 'E';
        header[12] = 'f'; header[13] = 'm'; header[14] = 't'; header[15] = ' ';
        header[16] = 16; header[17] = 0; header[18] = 0; header[19] = 0;
        header[20] = 1; header[21] = 0; // PCM format
        header[22] = (byte) channels; header[23] = 0;
        header[24] = (byte) (sampleRate & 0xff);
        header[25] = (byte) ((sampleRate >> 8) & 0xff);
        header[26] = (byte) ((sampleRate >> 16) & 0xff);
        header[27] = (byte) ((sampleRate >> 24) & 0xff);
        header[28] = (byte) (byteRate & 0xff);
        header[29] = (byte) ((byteRate >> 8) & 0xff);
        header[30] = (byte) ((byteRate >> 16) & 0xff);
        header[31] = (byte) ((byteRate >> 24) & 0xff);
        header[32] = (byte) (channels * (bitsPerSample / 8)); header[33] = 0;
        header[34] = (byte) bitsPerSample; header[35] = 0;
        header[36] = 'd'; header[37] = 'a'; header[38] = 't'; header[39] = 'a';
        header[40] = (byte) (audioLength & 0xff);
        header[41] = (byte) ((audioLength >> 8) & 0xff);
        header[42] = (byte) ((audioLength >> 16) & 0xff);
        header[43] = (byte) ((audioLength >> 24) & 0xff);

        out.write(header, 0, 44);
    }

    public void stop() {
        mIsStreaming = false;
        if (mAudioTrack != null) {
            try {
                mAudioTrack.pause();
                mAudioTrack.flush();
            } catch (IllegalStateException ignored) {}
        }
    }

    public enum Mood {
        NEUTRAL(0),
        HAPPY(1),
        SERIOUS(2),
        STORYTELLING(3),
        CALM(4),
        EXCITED(5),
        CASUAL_SLANG(6);

        public final int id;
        Mood(int id) { this.id = id; }
    }

    public void setMood(Mood mood) {
        if (mEngineHandle != 0 && mood != null) {
            nativeSetMood(mEngineHandle, mood.id);
        }
    }

    /**
     * Instantly speak text in any of the 21 languages with the specified Mood/Emotion
     * using the single locked voice (universal_speaker.bin).
     */
    public boolean speak(String text, String langCode, Mood mood) {
        setMood(mood);
        return speak(text, langCode);
    }

    public void release() {
        stop();
        if (mAudioTrack != null) {
            mAudioTrack.release();
            mAudioTrack = null;
        }
        if (mEngineHandle != 0) {
            nativeRelease(mEngineHandle);
            mEngineHandle = 0;
        }
    }

    // Native JNI Methods
    private native long nativeInit(String modelsDir, String speakerLockPath, boolean useVulkan, int numThreads);
    private native boolean nativeSynthesizeStream(long enginePtr, String text, String langCode, AudioChunkListener listener);
    private native byte[] nativeSynthesizeWav(long enginePtr, String text, String langCode);
    private native void nativeSetMood(long enginePtr, int moodId);
    private native void nativeRelease(long enginePtr);
}

