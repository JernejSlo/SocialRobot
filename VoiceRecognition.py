import whisper_timestamped as whisper
import pyaudio
import numpy as np
import re

# Load Whisper model (small is a good balance between speed and accuracy)
model = whisper.load_model("small")

# Audio settings
FORMAT = pyaudio.paInt16  # 16-bit format
CHANNELS = 1  # Mono audio
RATE = 16000  # Whisper works best at 16kHz
CHUNK = 1024  # Buffer size (number of frames per chunk)
RECORD_SECONDS = 5  # Duration to record

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open the microphone stream
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
print("Recording... Speak now!")

# Record audio
frames = []
for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):  # Loop for the duration
    data = stream.read(CHUNK)
    frames.append(np.frombuffer(data, dtype=np.int16))  # Convert to NumPy array

print("Recording stopped. Transcribing...")

# Close stream
stream.stop_stream()
stream.close()
audio.terminate()

# Convert recorded datasets into a NumPy array
audio_data = np.concatenate(frames, axis=0).astype(np.float32) / 32768.0  # Normalize

# Transcribe using Whisper
result = model.transcribe(audio_data, language="en")
print(result)
print("Transcription:", result["text"])

def extract_keywords_in_order(keywords, text):
    words = re.findall(r'\b\w+\b', text.lower())

    keyword_set = set(map(str.lower, keywords))
    extracted = [word for word in words if word.lower() in keyword_set]  # Keep only keywords
    return extracted

# Example usage
keywords = ["lemon","apple","banana","grab","hold","drop","give", "me", "hand", "find"]


result = extract_keywords_in_order(keywords, result["text"])
print(result)


