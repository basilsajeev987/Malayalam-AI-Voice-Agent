# Malayalam-AI-Voice-Agent
# 🎙️ Malayalam AI Voice Agent

A real-time Malayalam AI voice assistant that combines **Malayalam Speech-to-Text (STT)**, **Qwen3-8B**, and **Indic-Speak Text-to-Speech (TTS)** to create a natural conversational voice experience.

The project is designed to understand spoken Malayalam, generate contextual Malayalam responses, and convert those responses back into natural Malayalam speech.

---

## 🚀 Project Overview

The system follows this pipeline:

```text
              🎤 User Speech
                    │
                    ▼
          ┌─────────────────────┐
          │  Browser Microphone │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Speech-End Detection│
          │   Browser RMS VAD  │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Malayalam Whisper   │
          │   BuzzASR/malayalam │
          └──────────┬──────────┘
                     │
                     ▼
              Malayalam Text
                     │
                     ▼
          ┌─────────────────────┐
          │      Qwen3-8B       │
          │    LLM Response     │
          └──────────┬──────────┘
                     │
                     ▼
              Malayalam Text
                     │
                     ▼
          ┌─────────────────────┐
          │    Indic-Speak      │
          │   Malayalam TTS     │
          └──────────┬──────────┘
                     │
                     ▼
                🔊 Audio
```

---

# ✨ Features

- 🎤 Malayalam speech input
- 📝 Malayalam speech-to-text
- 🧠 Context-aware AI conversations
- 🤖 Qwen3-8B language model
- 🔊 Malayalam neural text-to-speech
- 👤 Malayalam voices including **Kiran** and **Lakshmi**
- 🗣️ Natural conversational responses
- ⏱️ Browser-based speech-end detection
- ⚡ Streaming LLM generation
- 🔄 Sentence-level TTS processing
- 🎧 Browser audio playback
- 💬 Conversation history
- 🛑 Voice-based exit commands
- 🚀 Designed for low-latency voice interaction
- 🔮 Future support for emotion-aware responses and barge-in

---

# 🧠 Models

## 1. Qwen3-8B

**Purpose:** Natural-language understanding and response generation.

```text
Model:
Qwen/Qwen3-8B
```

The model generates Malayalam responses using a custom system prompt designed for conversational speech.

The assistant is instructed to:

- Speak naturally in Malayalam
- Keep responses concise
- Avoid Markdown
- Avoid bullet points
- Maintain conversation context
- Ask follow-up questions when appropriate

---

## 2. BuzzASR Malayalam

**Purpose:** Malayalam Speech-to-Text.

```text
Model:
BuzzASR/malayalam
```

The model converts microphone audio into Malayalam text before sending it to Qwen3-8B.

Audio is converted to:

```text
16 kHz
Mono
Float32
```

before transcription.

---

## 3. Indic-Speak

**Purpose:** Malayalam Text-to-Speech.

```text
Model:
bodhan-ai/indic-speak
```

Malayalam speakers currently used:

```text
Kiran
Lakshmi
```

Example configuration:

```python
wav = tts(
    text,
    speaker="Kiran",
    style="NEUTRAL",
    temperature=0.45,
    top_p=0.90,
    top_k=40,
    seed=42,
    max_new_tokens=1000
)
```

---

# 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Environment | Google Colab |
| GPU | NVIDIA A100 40GB |
| LLM | Qwen3-8B |
| STT | BuzzASR Malayalam |
| TTS | Indic-Speak |
| Audio | FFmpeg |
| Browser Recording | JavaScript MediaRecorder |
| Speech Detection | Browser RMS-based detection |
| ML Framework | PyTorch |
| Model Hub | Hugging Face |

---

# 📁 Project Structure

Recommended structure:

```text
Malayalam-AI-Voice-Agent/
│
├── README.md
│
├── notebooks/
│   └── Malayalam_Voice_Agent.ipynb
│
├── models/
│   ├── Qwen3-8B/
│   ├── BuzzASR-malayalam/
│   └── indic-speak/
│
├── audio/
│   ├── input.wav
│   └── responses/
│
├── src/
│   ├── speech_to_text.py
│   ├── text_generation.py
│   ├── text_to_speech.py
│   ├── audio_recorder.py
│   └── voice_agent.py
│
└── requirements.txt
```

For Google Colab, the models can instead be stored under:

```text
/content/Malayalam_Voice_Agent_Models/
```

---

# ⚙️ Installation

Install the required packages:

```bash
pip install -U transformers
pip install -U huggingface_hub
pip install soundfile
pip install librosa
```

FFmpeg:

```bash
apt-get update
apt-get install -y ffmpeg
```

---

# 🔐 Hugging Face Authentication

Some models may require Hugging Face authentication.

```python
from huggingface_hub import login

login()
```

Enter your Hugging Face access token when prompted.

---

# 📥 Download Models

Create the model directory:

```python
import os

BASE_DIR = "/content/Malayalam_Voice_Agent_Models"

os.makedirs(BASE_DIR, exist_ok=True)
```

Download Qwen:

```python
from huggingface_hub import snapshot_download

QWEN_DIR = f"{BASE_DIR}/Qwen3-8B"

snapshot_download(
    repo_id="Qwen/Qwen3-8B",
    local_dir=QWEN_DIR
)
```

Download Malayalam STT:

```python
WHISPER_DIR = f"{BASE_DIR}/BuzzASR-malayalam"

snapshot_download(
    repo_id="BuzzASR/malayalam",
    local_dir=WHISPER_DIR
)
```

Download Indic-Speak:

```python
INDIC_DIR = f"{BASE_DIR}/indic-speak"

snapshot_download(
    repo_id="bodhan-ai/indic-speak",
    local_dir=INDIC_DIR
)
```

---

# 🧠 Loading Qwen3-8B

```python
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

qwen_tokenizer = AutoTokenizer.from_pretrained(
    QWEN_DIR,
    local_files_only=True
)

qwen_model = AutoModelForCausalLM.from_pretrained(
    QWEN_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)

qwen_model.eval()
```

---

# 🎤 Loading Malayalam STT

```python
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq
)

whisper_processor = AutoProcessor.from_pretrained(
    WHISPER_DIR,
    local_files_only=True
)

whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    WHISPER_DIR,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    local_files_only=True
).to("cuda")

whisper_model.eval()
```

---

# 🔊 Loading Indic-Speak

```python
import sys

sys.path.insert(0, INDIC_DIR)

from inference import TTS

tts = TTS(INDIC_DIR)
```

---

# 📝 Speech-to-Text

The microphone recording is converted to 16 kHz mono WAV audio.

```python
import soundfile as sf
import numpy as np
import torch

def transcribe_malayalam(audio_file):

    audio, sr = sf.read(audio_file)

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    if audio.ndim > 1:
        audio = np.mean(
            audio,
            axis=1
        )

    if sr != 16000:

        import librosa

        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=16000
        )

        sr = 16000

    inputs = whisper_processor(
        audio=audio,
        sampling_rate=sr,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(
        "cuda",
        dtype=torch.float16
    )

    with torch.inference_mode():

        generated_ids = whisper_model.generate(
            input_features=input_features,
            max_new_tokens=64,
            num_beams=1,
            do_sample=False
        )

    text = whisper_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()

    return text
```

---

# 🤖 Malayalam AI Response

The assistant uses a Malayalam-specific system prompt.

```python
SYSTEM_PROMPT = """
നിങ്ങൾ ഒരു സ്വാഭാവിക മലയാളം AI വോയ്സ് അസിസ്റ്റന്റാണ്.

മനുഷ്യനോട് സംസാരിക്കുന്നതുപോലെ സ്വാഭാവികമായി സംസാരിക്കുക.

എപ്പോഴും മലയാളത്തിൽ മറുപടി നൽകുക.

മറുപടികൾ ചെറുതും സംസാരിക്കാൻ എളുപ്പമുള്ളതുമായിരിക്കണം.
സാധാരണയായി 2 മുതൽ 5 വരെ വാക്യങ്ങൾ നൽകുക.

വളരെ ഔപചാരികമായ ഭാഷ ഒഴിവാക്കുക.
സ്വാഭാവികമായ മലയാളം ഉപയോഗിക്കുക.

Markdown ഉപയോഗിക്കരുത്.
Bullet points ഉപയോഗിക്കരുത്.

ഉപയോക്താവ് ചോദിച്ച കാര്യത്തിന് നേരിട്ട് മറുപടി നൽകുക.

ആവശ്യമുള്ളപ്പോൾ ചെറിയ follow-up ചോദ്യം ചോദിക്കാം.

മുൻ സംഭാഷണത്തിന്റെ context ഓർമ്മയിൽ സൂക്ഷിക്കുക.
"""
```

---

# 🔊 Malayalam TTS

```python
SPEAKER = "Kiran"

def generate_tts(text):

    text = str(text).strip()

    text = text.replace("*", "")
    text = text.replace("#", "")
    text = text.replace("•", "")

    text = " ".join(
        text.split()
    )

    wav = tts(
        text,
        speaker=SPEAKER,
        style="NEUTRAL",
        temperature=0.45,
        top_p=0.90,
        top_k=40,
        seed=42,
        max_new_tokens=1000
    )

    output_file = (
        "/content/voice_response.wav"
    )

    tts.save(
        output_file,
        wav
    )

    return output_file
```

---

# ⚡ Streaming Architecture

The project supports sentence-level streaming.

Instead of waiting for the entire response:

```text
User
 ↓
Whisper
 ↓
Qwen starts generating
 ↓
Sentence 1
 ↓
TTS Sentence 1
 ↓
🔊 Play
 ↓
Sentence 2
 ↓
TTS Sentence 2
 ↓
🔊 Play
```

This reduces the perceived response latency.

---

# 🧵 Concurrent Processing

The architecture uses queues:

```text
                  Qwen
                   │
                   ▼
             Sentence Queue
                   │
                   ▼
               TTS Worker
                   │
                   ▼
              Audio Queue
                   │
                   ▼
               Browser
```

This allows Qwen generation and TTS processing to operate concurrently.

---

# 🎤 Automatic Speech-End Detection

The browser continuously monitors microphone audio.

The current implementation uses an RMS threshold:

```text
Microphone
     ↓
Audio analyser
     ↓
RMS calculation
     ↓
Speech detected
     ↓
User stops speaking
     ↓
Silence ~0.8 seconds
     ↓
Recording ends
```

This provides a more natural experience than recording a fixed number of seconds.

---

# 🗣️ Conversation Memory

Conversation history is maintained using:

```python
conversation_history = []
```

Messages are stored as:

```python
{
    "role": "user",
    "content": "..."
}
```

and:

```python
{
    "role": "assistant",
    "content": "..."
}
```

The most recent conversation turns are supplied to Qwen3-8B.

---

# 🛑 Voice Exit Commands

The assistant can terminate when the user says commands such as:

```text
exit
quit
stop
നിർത്തുക
നിർത്തൂ
പോകട്ടെ
```

---

# ▶️ Running the Voice Agent

The complete system runs as:

```python
run_streaming_voice_agent()
```

The interaction becomes:

```text
🎤 Listening...

User speaks Malayalam

        ↓

📝 Malayalam transcription

        ↓

🤖 Qwen generates response

        ↓

🔊 Indic-Speak generates speech

        ↓

🎧 Browser plays response

        ↓

🔄 Listen again
```

---

# 🔮 Future Development

The project is intended to evolve into a more advanced real-time conversational AI system.

## Phase 1 — Core Voice Agent

- [x] Malayalam STT
- [x] Qwen3-8B
- [x] Malayalam TTS
- [x] Conversation history
- [x] Automatic speech-end detection
- [x] Sentence-level streaming
- [x] Browser audio playback

## Phase 2 — Real-Time Interaction

- [ ] True Silero VAD
- [ ] Lower STT latency
- [ ] Lower TTS latency
- [ ] Audio streaming instead of complete WAV files
- [ ] Improved browser audio playback
- [ ] Real-time microphone pipeline

## Phase 3 — Barge-In

Target behaviour:

```text
AI is speaking 🔊
       ↓
User starts speaking 🎤
       ↓
Detect user speech
       ↓
STOP AI AUDIO
       ↓
Listen to user
       ↓
Whisper
       ↓
Qwen
       ↓
TTS
```

This will allow users to interrupt the assistant naturally.

## Phase 4 — Emotion-Aware AI

Future versions will incorporate emotion detection.

Potential pipeline:

```text
User Voice
    │
    ├───────────────┐
    ▼               ▼
   STT          Emotion Model
    │               │
    └───────┬───────┘
            ▼
         Qwen3-8B
            │
            ▼
       Emotion-aware
         response
            │
            ▼
       Indic-Speak
            │
            ▼
          🔊
```

The goal is to make the assistant respond differently depending on conversational context and detected emotion.

---

# 🌟 Long-Term Goal

The long-term objective is to build a **natural Malayalam conversational AI agent** capable of:

- Continuous conversations
- Natural turn-taking
- Malayalam understanding
- Context retention
- Emotional awareness
- Barge-in
- Low-latency streaming
- Expressive Malayalam speech
- Personalized voices
- Real-time interaction

The eventual architecture is intended to move from a Colab prototype toward a production-ready real-time voice system.

---

# 📌 Current Status

```text
                 PROJECT STATUS

Malayalam STT              ✅
Qwen3-8B                   ✅
Indic-Speak TTS            ✅
Kiran voice                ✅
Lakshmi voice              ✅
Conversation memory        ✅
Speech-end detection      ✅
Streaming Qwen             ✅
Concurrent TTS             ✅
Browser playback           ✅

True VAD                    🔄
Barge-in                    🔄
Emotion detection           🔄
Real-time audio streaming   🔄
Production deployment       🔜
```

---

# 👨‍💻 Project

**Malayalam AI Voice Agent**

Built with:

```text
Python
PyTorch
Qwen3-8B
BuzzASR Malayalam
Indic-Speak
Transformers
Hugging Face
Google Colab
NVIDIA A100
```

---

## License & Model Attribution

The individual AI models used by this project are provided by their respective authors and organizations. Check the original model repositories and their licenses before using the models commercially or redistributing model weights.

This repository contains the voice-agent integration and application logic rather than claiming ownership of the underlying pretrained models.
