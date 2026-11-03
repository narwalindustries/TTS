#pragma once

/**
 * Lightweight Real-Time Audio Output Engine for HDModularTTS
 * Direct hardware audio streaming for Windows (WaveOut/WASAPI)
 * Zero external audio dependencies required.
 */

#include <vector>
#include <cstdint>
#include <iostream>
#include <cstring>

#if defined(_WIN32) || defined(_WIN64)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
    #include <mmsystem.h>
    #pragma comment(lib, "winmm.lib")
#endif

namespace AudioDevice {

class RealtimeAudioPlayer {
public:
    RealtimeAudioPlayer(int sample_rate = 24000, int channels = 1)
        : m_sample_rate(sample_rate), m_channels(channels) {
#if defined(_WIN32) || defined(_WIN64)
        WAVEFORMATEX wfx;
        std::memset(&wfx, 0, sizeof(WAVEFORMATEX));
        wfx.wFormatTag = WAVE_FORMAT_PCM;
        wfx.nChannels = static_cast<WORD>(m_channels);
        wfx.nSamplesPerSec = static_cast<DWORD>(m_sample_rate);
        wfx.wBitsPerSample = 16;
        wfx.nBlockAlign = (wfx.nChannels * wfx.wBitsPerSample) / 8;
        wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign;

        MMRESULT res = waveOutOpen(&m_hWaveOut, WAVE_MAPPER, &wfx, 0, 0, CALLBACK_NULL);
        if (res == MMSYSERR_NOERROR) {
            m_is_opened = true;
        } else {
            std::cerr << "[AudioPlayer] Warning: Could not open default audio device (error " << res << ")" << std::endl;
        }
#endif
    }

    ~RealtimeAudioPlayer() {
        stop();
    }

    void playChunk(const std::vector<float>& float_samples) {
        if (float_samples.empty()) return;

        // Convert float32 [-1.0, 1.0] to int16 PCM
        std::vector<int16_t> pcm16(float_samples.size());
        for (size_t i = 0; i < float_samples.size(); ++i) {
            float s = std::max(-1.0f, std::min(1.0f, float_samples[i]));
            pcm16[i] = static_cast<int16_t>(s * 32767.0f);
        }

#if defined(_WIN32) || defined(_WIN64)
        if (m_is_opened && m_hWaveOut) {
            size_t bytes = pcm16.size() * sizeof(int16_t);
            WAVEHDR* header = new WAVEHDR();
            std::memset(header, 0, sizeof(WAVEHDR));
            char* buf = new char[bytes];
            std::memcpy(buf, pcm16.data(), bytes);
            header->lpData = buf;
            header->dwBufferLength = static_cast<DWORD>(bytes);

            waveOutPrepareHeader(m_hWaveOut, header, sizeof(WAVEHDR));
            waveOutWrite(m_hWaveOut, header, sizeof(WAVEHDR));

            m_headers.push_back(header);
        }
#endif
    }

    void stop() {
#if defined(_WIN32) || defined(_WIN64)
        if (m_is_opened && m_hWaveOut) {
            waveOutReset(m_hWaveOut);
            for (auto* hdr : m_headers) {
                waveOutUnprepareHeader(m_hWaveOut, hdr, sizeof(WAVEHDR));
                delete[] hdr->lpData;
                delete hdr;
            }
            m_headers.clear();
            waveOutClose(m_hWaveOut);
            m_hWaveOut = NULL;
            m_is_opened = false;
        }
#endif
    }

    static bool writeWavFile(const std::string& filepath, const std::vector<float>& samples, int sample_rate = 24000) {
        FILE* fp = fopen(filepath.c_str(), "wb");
        if (!fp) return false;

        uint32_t num_samples = static_cast<uint32_t>(samples.size());
        uint32_t data_chunk_size = num_samples * 2; // 16-bit mono
        uint32_t file_chunk_size = 36 + data_chunk_size;
        uint16_t num_channels = 1;
        uint16_t bits_per_sample = 16;
        uint32_t byte_rate = sample_rate * num_channels * (bits_per_sample / 8);
        uint16_t block_align = num_channels * (bits_per_sample / 8);

        // RIFF header
        fwrite("RIFF", 1, 4, fp);
        fwrite(&file_chunk_size, 4, 1, fp);
        fwrite("WAVE", 1, 4, fp);

        // fmt subchunk
        fwrite("fmt ", 1, 4, fp);
        uint32_t fmt_size = 16;
        uint16_t audio_format = 1; // PCM
        fwrite(&fmt_size, 4, 1, fp);
        fwrite(&audio_format, 2, 1, fp);
        fwrite(&num_channels, 2, 1, fp);
        fwrite(&sample_rate, 4, 1, fp);
        fwrite(&byte_rate, 4, 1, fp);
        fwrite(&block_align, 2, 1, fp);
        fwrite(&bits_per_sample, 2, 1, fp);

        // data subchunk
        fwrite("data", 1, 4, fp);
        fwrite(&data_chunk_size, 4, 1, fp);

        // PCM 16-bit samples
        for (float s : samples) {
            float clamped = std::max(-1.0f, std::min(1.0f, s));
            int16_t pcm = static_cast<int16_t>(clamped * 32767.0f);
            fwrite(&pcm, 2, 1, fp);
        }

        fclose(fp);
        return true;
    }

private:
    int m_sample_rate{24000};
    int m_channels{1};
    bool m_is_opened{false};

#if defined(_WIN32) || defined(_WIN64)
    HWAVEOUT m_hWaveOut{NULL};
    std::vector<WAVEHDR*> m_headers;
#endif
};

} // namespace AudioDevice

