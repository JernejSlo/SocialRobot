import whisper_timestamped as whisper
import pyaudio
import numpy as np
import re
import keyboard
import time

class VoiceRecognitionUtils():
    def __init__(self, model_size="small", keywords=None):
        self.vr_model = whisper.load_model(model_size)
        self.keywords = keywords or ["lemon", "apple", "banana", "grab", "hold", "drop", "give", "me", "hand", "find"]
        self.word_buffer = []
        self.silence_loops = 0
        self.max_silence_loops = 10

        self.FORMAT = pyaudio.paInt16  # 16-bit format
        self.CHANNELS = 1  # Mono audio
        self.RATE = 16000  # Whisper works best at 16kHz
        self.CHUNK = 1024  # Buffer size (number of frames per chunk)
        self.RECORD_SECONDS = 5  # Duration to record

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Open the microphone stream
        self.stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)

    def list_input_devices(self):
        audio = pyaudio.PyAudio()
        print("🎤 Available input devices:")

        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                print(
                    f"[{i}] {info['name']} - {int(info['maxInputChannels'])} channels - {int(info['defaultSampleRate'])} Hz")

        audio.terminate()

    def extract_keywords_in_order(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        keyword_set = set(map(str.lower, self.keywords))
        return [word for word in words if word.lower() in keyword_set]



    def listen(self):
        frames = []
        print("Listening. ")
        for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = self.stream.read(self.CHUNK)
            frames.append(np.frombuffer(data, dtype=np.int16))

        audio_data = np.concatenate(frames, axis=0).astype(np.float32) / 32768.0
        result = self.vr_model.transcribe(audio_data, language="en")
        text = result.get("text", "")
        print(f"📝 Heard: {text}")

        return self.extract_keywords_in_order(text)


    def cleanup(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        print("🔊 Stream closed.")


#VoiceRecognitionUtils().list_input_devices()