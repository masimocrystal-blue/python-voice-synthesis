# ClaudePython — formant speech synthesis from scratch

English speech synthesized in pure Python (NumPy / SciPy) with a rule-based
Klatt-style formant synthesizer. No TTS engine, trained model, OS speech
feature, external API, or recorded audio is used.

## Files

| File | Contents |
|---|---|
| `speech.wav` | Main result: ~10 s self-introduction (22,050 Hz, 16-bit mono) |
| `spectrogram.png` | Spectrogram of `speech.wav` with phoneme labels |
| `demo_other_sentences.wav` / `.png` | Same code, different sentences |
| `formant_tts.py` | The synthesizer (the only code used to make the audio) |
| `verify.py` | Analysis only: LPC formant check, consonant levels, optional ASR test |
| `process/` | Intermediate outputs from the tuning process (see below) |

Spoken text in `speech.wav`:
"Hello. I am Claude, an AI made by Anthropic. This voice is not a recording.
Every sample was computed from scratch in Python."

## Usage

```
pip install numpy scipy matplotlib
python formant_tts.py                                  # default text -> speech.wav
python formant_tts.py "My name is Claude." -o out.wav --spectrogram out.png
python formant_tts.py "Hello {W ER1 L D}."             # ARPABET in braces for any word
```

Options: `--rate` (speaking rate, default 0.9), `--f0` (base pitch in Hz, default 112).

## What text it can say

- Any sequence of the 40 ARPABET phonemes can be synthesized.
- Plain text is limited to the 74 words in the built-in `LEXICON`; there is no
  spelling-to-sound rule set, so unknown words raise an error.
- Unknown words can be written as phonemes in braces, e.g. `{B AH0 N AE1 N AH0}`.

## process/

- `v1.wav`, `v1.png` — first version (bright noise bands, S and SH not distinguished)
- `fox_v2.png` — test sentences before the consonant-noise rewrite
- `fox_v3.png` — test sentences after switching to FIR-shaped noise and adding F7–F9
