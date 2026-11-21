ClaudeDS10 - KORG DS-10 style speech experiment
================================================
Spoken text:
  "Hello. I am Claude. I am an A.I. Now I live in a KORG D S ten.
   I only have two voices. Nice to meet you."

Files
  ds10_talk.wav               final audio (44.1 kHz / 16-bit / stereo, 11.52 s)
  ds10_talk.py                synth engine + sequencer + patch + script (run: python3 ds10_talk.py)
  ds10_talk_sequence.txt      all 12 patterns x 16 steps (NOTE/GATE/VOL/PAN/KAOSS X/Y), readable
  ds10_talk_sequence.json     same data + SYN1/SYN2 patch settings, machine-readable
  ds10_talk_spectrogram.png   spectrogram of the final render with phoneme labels

Setup
  BPM 250 (1 step = 16th = 60 ms), SYN1 + SYN2 only, no DRUM, FX off.
  SYN1: SAW/NOISE -> LPF (F1).  SYN2: SAW/NOISE -> BPF (F2 / frication).
  KAOSS X = EG INT (SMOOTH) = formant frequency, KAOSS Y = VCO BALANCE (HOLD) = voiced/noise.
  Requires numpy, scipy (numba optional, for speed).
