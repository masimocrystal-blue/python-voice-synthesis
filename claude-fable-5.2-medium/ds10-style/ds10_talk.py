#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ds10_talk.py -- "KORG DS-10 shibari" speech experiment.

A DS-10-style engine (SYN1 + SYN2 only, no drums) plus a 16-step pattern
sequencer with KAOSS X/Y lanes, used to speak an English sentence.

Run:  python3 ds10_talk.py          -> ds10_talk.wav, ds10_talk_sequence.json
Needs numpy, scipy.  numba is optional (only makes the per-sample loop fast).

Everything that makes sound is one of:
  VCO1/VCO2 (tri/saw/square/noise), balance, VCO2 pitch, sync/ring,
  ONE VCF per synth (LPF/HPF/BPF, cutoff, peak, EG int), ONE ADSR EG,
  VCA (level, drive, EG/GATE), MG (LFO) + patch matrix, portamento,
  master FX (delay / chorus / flanger, off in this patch).
Everything that moves over time is one of:
  NOTE / GATE / VOLUME / PAN lanes (one value per 16th-note step),
  KAOSS X and KAOSS Y lanes (one 12-level value per step, HOLD or SMOOTH),
  EG, MG, portamento.
"""
import json
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter

try:
    from numba import njit
except Exception:                                   # pure-python fallback (slow)
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (len(a) == 1 and callable(a[0])) else a[0]

FS = 44100
KNOB_MAX = 127          # all knobs are 0..127 integers
KAOSS_LEVELS = 12       # recorded KAOSS value per step: 0..11
VOL_LEVELS = 12         # VOLUME lane per step: 0..11
PAN_POS = (-1.0, -0.5, 0.0, 0.5, 1.0)   # L100 L50 C R50 R100

# ----------------------------------------------------------------- knob laws
def knob_cutoff_hz(k):   return 20.0 * 2.0 ** (10.0 * k / KNOB_MAX)   # 20 Hz..20.5 kHz
def knob_peak_q(k):      return 0.5 + 24.5 * (k / KNOB_MAX) ** 2      # Q 0.5..25
def knob_egint_oct(k):   return 10.0 * k / KNOB_MAX                   # octaves at EG=1
def knob_time_s(k):      return 0.0 if k == 0 else 0.001 * (5000.0 ** (k / KNOB_MAX))
def knob_porta_s(k):     return 0.0 if k == 0 else 0.002 * (250.0 ** (k / KNOB_MAX))
def knob_mg_hz(k):       return 0.1 * (300.0 ** (k / KNOB_MAX))       # 0.1..30 Hz
def knob_vco2_semi(k):   return (k - 64) / 64.0 * 12.0                # +-1 octave
def kaoss_to_knob(level): return int(round(level * KNOB_MAX / (KAOSS_LEVELS - 1)))

WAVES = {"TRI": 0, "SAW": 1, "SQU": 2, "NOISE": 3}
VCF_TYPES = {"LPF": 0, "HPF": 1, "BPF": 2}
SYNC_MODES = {"OFF": 0, "SYNC": 1, "RING": 2}
PATCH_SRC = {"MG_TRI": 0, "MG_SAW": 1, "MG_SQU": 2, "MG_SH": 3, "EG": 4, "VCO2": 5}
PATCH_DST = {"PITCH": 0, "VCO2_PITCH": 1, "CUTOFF": 2, "VCA": 3}
KAOSS_TARGETS = ("CUTOFF", "PEAK", "EG_INT", "BALANCE", "VCO2_PITCH", "DRIVE", "LEVEL")

# ----------------------------------------------------------------- DSP core
@njit(cache=True)
def _blep(t, dt):
    if t < dt:
        x = t / dt
        return x + x - x * x - 1.0
    if t > 1.0 - dt:
        x = (t - 1.0) / dt
        return x * x + x + x + 1.0
    return 0.0


@njit(cache=True)
def _osc(wave, ph, dt, nz):
    if wave == 1:                       # saw
        return 2.0 * ph - 1.0 - _blep(ph, dt)
    if wave == 2:                       # square
        v = 1.0 if ph < 0.5 else -1.0
        v += _blep(ph, dt)
        p2 = ph + 0.5
        if p2 >= 1.0:
            p2 -= 1.0
        return v - _blep(p2, dt)
    if wave == 0:                       # triangle
        return 4.0 * abs(ph - 0.5) - 1.0
    return nz                           # white noise


@njit(cache=True)
def _core(fs, f0, gate, trig, vol, cutoff_hz, q, egint_oct, bal, v2semi, drive, level,
          wave1, wave2, sync_mode, vcf_type, tA, tD, sus, tR, vca_eg, mg_hz, P, noise):
    n = f0.shape[0]
    out = np.zeros(n)
    ph1 = 0.0
    ph2 = 0.0
    eg = 0.0
    stage = 0                           # 0 idle/release, 1 attack, 2 decay/sustain
    ic1 = 0.0
    ic2 = 0.0
    mgph = 0.0
    sh = 0.0
    vca_s = 0.0                         # de-click slew on VCA gain (2 ms)
    slew = 1.0 - np.exp(-1.0 / (0.002 * fs))
    cD = np.exp(-1.0 / (max(tD, 1e-4) * fs / 4.0))
    cR = np.exp(-1.0 / (max(tR, 1e-4) * fs / 4.0))
    v2 = 0.0
    for i in range(n):
        # ---- MG
        mgph += mg_hz / fs
        if mgph >= 1.0:
            mgph -= 1.0
            sh = noise[i]
        mg_tri = 4.0 * abs(mgph - 0.5) - 1.0
        mg_saw = 2.0 * mgph - 1.0
        mg_squ = 1.0 if mgph < 0.5 else -1.0
        # ---- EG (ADSR)
        if trig[i] > 0.5:
            stage = 1
        if gate[i] < 0.5:
            stage = 0
        if stage == 1:
            if tA <= 0.0:
                eg = 1.0
            else:
                eg += 1.0 / (tA * fs)
            if eg >= 1.0:
                eg = 1.0
                stage = 2
        if stage == 2:
            if tD <= 0.0:
                eg = sus
            else:
                eg = sus + (eg - sus) * cD
        if stage == 0:
            if tR <= 0.0:
                eg = 0.0
            else:
                eg *= cR
        # ---- patch sources
        src0 = mg_tri; src1 = mg_saw; src2 = mg_squ; src3 = sh; src4 = eg; src5 = v2
        m_pitch = P[0, 0]*src0 + P[1, 0]*src1 + P[2, 0]*src2 + P[3, 0]*src3 + P[4, 0]*src4 + P[5, 0]*src5
        m_v2p   = P[0, 1]*src0 + P[1, 1]*src1 + P[2, 1]*src2 + P[3, 1]*src3 + P[4, 1]*src4 + P[5, 1]*src5
        m_cut   = P[0, 2]*src0 + P[1, 2]*src1 + P[2, 2]*src2 + P[3, 2]*src3 + P[4, 2]*src4 + P[5, 2]*src5
        m_vca   = P[0, 3]*src0 + P[1, 3]*src1 + P[2, 3]*src2 + P[3, 3]*src3 + P[4, 3]*src4 + P[5, 3]*src5
        # ---- VCOs
        fr1 = f0[i] * 2.0 ** (m_pitch)                    # patch amount 1.0 = 1 octave
        fr2 = fr1 * 2.0 ** (v2semi[i] / 12.0 + m_v2p)
        dt1 = fr1 / fs
        dt2 = fr2 / fs
        ph1 += dt1
        wrapped = False
        if ph1 >= 1.0:
            ph1 -= 1.0
            wrapped = True
        ph2 += dt2
        if ph2 >= 1.0:
            ph2 -= 1.0
        if sync_mode == 1 and wrapped:
            ph2 = ph1 * dt2 / dt1
        v1 = _osc(wave1, ph1, dt1, noise[i])
        v2 = _osc(wave2, ph2, dt2, noise[i])
        if sync_mode == 2:
            v2 = v1 * v2
        b = bal[i]
        x = (1.0 - b) * v1 + b * v2
        # ---- VCF (one 12 dB/oct state-variable filter)
        fc = cutoff_hz[i] * 2.0 ** (egint_oct[i] * eg + 5.0 * m_cut)
        if fc > 0.45 * fs:
            fc = 0.45 * fs
        if fc < 10.0:
            fc = 10.0
        g = np.tan(np.pi * fc / fs)
        k = 1.0 / q[i]
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v3 = x - ic2
        bp = a1 * ic1 + a2 * v3
        lp = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * bp - ic1
        ic2 = 2.0 * lp - ic2
        if vcf_type == 0:
            y = lp
        elif vcf_type == 1:
            y = x - k * bp - lp
        else:
            y = bp
        # ---- VCA
        amp = eg if vca_eg else gate[i]
        amp = amp * vol[i] * (1.0 + m_vca)
        if amp < 0.0:
            amp = 0.0
        vca_s += slew * (amp - vca_s)
        dg = 1.0 + 7.0 * drive[i]
        y = np.tanh(y * dg) / dg ** 0.5
        out[i] = y * vca_s * level[i]
    return out


# ----------------------------------------------------------------- synth + sequencer
class Synth:
    """One DS-10 synth voice. `patch` is a dict of knob values (0..127) and switches."""

    def __init__(self, name, patch):
        self.name = name
        self.p = patch

    def render(self, steps, bpm, seed):
        """steps: list of dicts {note, gate, vol, pan, x, y}. gate in (0,25,50,75,100,'L')."""
        p = self.p
        spb = 60.0 / bpm / 4.0                       # seconds per 16th step
        ns = int(round(spb * FS))
        n = ns * len(steps)
        t_in = np.arange(ns) / ns
        semis = np.zeros(n); gate = np.zeros(n); trig = np.zeros(n)
        vol = np.zeros(n); pan = np.zeros(n)
        kx = np.zeros(n); ky = np.zeros(n)
        last_note = steps[0]["note"] if steps[0]["note"] is not None else 45
        prev_legato = False
        for s, st in enumerate(steps):
            a = s * ns
            if st["note"] is not None:
                last_note = st["note"]
            semis[a:a + ns] = last_note
            g = st["gate"]
            if g != 0:
                glen = ns if g in ("L", 100) else int(ns * g / 100.0)
                gate[a:a + glen] = 1.0
                if not prev_legato:
                    trig[a] = 1.0
            prev_legato = (g == "L")
            vol[a:a + ns] = st["vol"] / (VOL_LEVELS - 1)
            pan[a:a + ns] = PAN_POS[st["pan"]]
            nxt = steps[min(s + 1, len(steps) - 1)]
            for arr, key, mode in ((kx, "x", p["KAOSS_X_MODE"]), (ky, "y", p["KAOSS_Y_MODE"])):
                v0 = kaoss_to_knob(st[key])
                if mode == "SMOOTH":                # linear glide to the next step's value
                    arr[a:a + ns] = v0 + (kaoss_to_knob(nxt[key]) - v0) * t_in
                else:                               # HOLD
                    arr[a:a + ns] = v0
        # portamento: one-pole glide on pitch
        tp = knob_porta_s(p["PORTAMENTO"])
        if tp > 0:
            c = np.exp(-1.0 / (tp * FS))
            semis = lfilter([1 - c], [1, -c], semis, zi=[semis[0] * c])[0]
        f0 = 440.0 * 2.0 ** ((semis - 69 + 12 * p["VCO1_OCTAVE"]) / 12.0)
        # knob arrays (static unless taken over by a KAOSS axis)
        knobs = {t: np.full(n, float(p[t])) for t in KAOSS_TARGETS}
        knobs[p["KAOSS_X"]] = kx
        knobs[p["KAOSS_Y"]] = ky
        P = np.zeros((6, 4))
        for src, dst, amt in p.get("PATCH", []):
            P[PATCH_SRC[src], PATCH_DST[dst]] = amt / KNOB_MAX
        noise = np.random.default_rng(seed).uniform(-1, 1, n)
        y = _core(float(FS), f0, gate, trig, vol,
                  knob_cutoff_hz(knobs["CUTOFF"]), knob_peak_q(knobs["PEAK"]),
                  knob_egint_oct(knobs["EG_INT"]), knobs["BALANCE"] / KNOB_MAX,
                  knob_vco2_semi(knobs["VCO2_PITCH"]), knobs["DRIVE"] / KNOB_MAX,
                  knobs["LEVEL"] / KNOB_MAX,
                  WAVES[p["VCO1_WAVE"]], WAVES[p["VCO2_WAVE"]], SYNC_MODES[p["VCO_SYNC"]],
                  VCF_TYPES[p["VCF_TYPE"]],
                  knob_time_s(p["EG_A"]), knob_time_s(p["EG_D"]), p["EG_S"] / KNOB_MAX,
                  knob_time_s(p["EG_R"]), p["VCA_MODE"] == "EG",
                  knob_mg_hz(p["MG_RATE"]), P, noise)
        th = (pan + 1.0) * np.pi / 4.0
        return np.stack([y * np.cos(th), y * np.sin(th)], axis=1) * np.sqrt(2.0)


def master_fx(x, fx, bpm):
    """DS-10 mixer FX: one of DELAY / CHORUS / FLANGER (or OFF)."""
    kind = fx.get("TYPE", "OFF")
    if kind == "OFF":
        return x
    n = x.shape[0]
    y = x.copy()
    if kind == "DELAY":
        d = int(60.0 / bpm / 4.0 * fx.get("TIME_STEPS", 3) * FS)
        fb, mix = fx.get("FEEDBACK", 0.3), fx.get("MIX", 0.25)
        buf = x.copy()
        for i in range(d, n):
            buf[i] += fb * buf[i - d]
        y[d:] += mix * buf[:-d]
        return y
    rate, depth, base = (0.6, 0.004, 0.015) if kind == "CHORUS" else (0.25, 0.002, 0.003)
    t = np.arange(n) / FS
    dl = (base + depth * np.sin(2 * np.pi * rate * t)) * FS
    idx = np.arange(n) - dl
    i0 = np.clip(np.floor(idx).astype(int), 0, n - 2)
    fr = np.clip(idx - i0, 0, 1)
    for c in range(2):
        y[:, c] = x[:, c] + fx.get("MIX", 0.5) * ((1 - fr) * x[i0, c] + fr * x[i0 + 1, c])
    return y


# ================================================================= THE PATCH
BPM = 250                                # 1 step = 60 ms

SYN1_PATCH = {                           # "F1 + voice bar": saw/noise -> resonant LPF
    "VCO1_WAVE": "SAW", "VCO1_OCTAVE": 0, "VCO2_WAVE": "NOISE", "VCO2_PITCH": 64,
    "VCO_SYNC": "OFF", "BALANCE": 0, "PORTAMENTO": 72,
    "VCF_TYPE": "LPF", "CUTOFF": 42, "PEAK": 45, "EG_INT": 0,
    "EG_A": 0, "EG_D": 0, "EG_S": 28, "EG_R": 127,
    "VCA_MODE": "GATE", "LEVEL": 100, "DRIVE": 0,
    "MG_RATE": 88, "PATCH": [("MG_TRI", "PITCH", 1)],
    "KAOSS_X": "EG_INT", "KAOSS_X_MODE": "SMOOTH",      # X = F1 frequency
    "KAOSS_Y": "BALANCE", "KAOSS_Y_MODE": "HOLD",       # Y = saw <-> noise
}
SYN2_PATCH = {                           # "F2 + frication": saw/noise -> resonant BPF
    "VCO1_WAVE": "SAW", "VCO1_OCTAVE": 0, "VCO2_WAVE": "NOISE", "VCO2_PITCH": 64,
    "VCO_SYNC": "OFF", "BALANCE": 0, "PORTAMENTO": 72,
    "VCF_TYPE": "BPF", "CUTOFF": 66, "PEAK": 75, "EG_INT": 0,
    "EG_A": 0, "EG_D": 0, "EG_S": 35, "EG_R": 127,
    "VCA_MODE": "GATE", "LEVEL": 100, "DRIVE": 0,
    "MG_RATE": 88, "PATCH": [("MG_TRI", "PITCH", 1)],
    "KAOSS_X": "EG_INT", "KAOSS_X_MODE": "SMOOTH",      # X = F2 / noise-band frequency
    "KAOSS_Y": "BALANCE", "KAOSS_Y_MODE": "HOLD",       # Y = saw <-> noise
}
FX = {"TYPE": "OFF"}


def x_grid(patch):
    """The 12 filter frequencies reachable by KAOSS X (= EG INT) for a patch."""
    base = knob_cutoff_hz(patch["CUTOFF"])
    s = patch["EG_S"] / KNOB_MAX
    return np.array([base * 2.0 ** (knob_egint_oct(kaoss_to_knob(L)) * s)
                     for L in range(KAOSS_LEVELS)])


GRID1, GRID2 = x_grid(SYN1_PATCH), x_grid(SYN2_PATCH)


def q_hz(grid, hz):
    return int(np.argmin(np.abs(np.log(grid / hz))))


# ================================================================= PHONEME -> STEPS
# frame = [F1 Hz, F2 Hz, vol1, vol2, bal1, bal2]   (vol, bal on the 0..11 lanes)
V = {  # vowels / sonorants: (F1, F2, vol1, vol2)
    "IY": (280, 2800, 9, 10), "IH": (400, 2150, 10, 10), "EH": (560, 1950, 10, 10),
    "AE": (700, 1800, 11, 10), "AA": (750, 1150, 11, 10), "AH": (620, 1250, 10, 9),
    "AX": (500, 1400, 8, 7),  "AO": (580, 900, 11, 10),  "UH": (440, 1050, 10, 9),
    "UW": (310, 900, 9, 8),   "O":  (480, 900, 10, 9),   "E":  (450, 2350, 10, 10),
    "L":  (380, 1000, 7, 4),  "R":  (380, 1250, 7, 5),   "W":  (300, 720, 7, 4),
    "Y":  (270, 2600, 7, 6),
    "M":  (250, 1000, 5, 2),  "N":  (250, 1550, 5, 2),
}
DIPH = {"AY": ("AA", "IH"), "EY": ("E", "IY"), "OW": ("O", "UW"),
        "AW": ("AA", "UH"), "OY": ("AO", "IH")}
S_HZ = 5000.0


def phon(sym, n, nxt_vowel):
    """Return n-ish frames for phoneme `sym`. nxt_vowel = (F1, F2) of the following vowel."""
    nf1, nf2 = nxt_vowel
    if sym in V:
        f1, f2, a, b = V[sym]
        return [[f1, f2, a, b, 0, 0] for _ in range(n)]
    if sym in DIPH:
        s, e = DIPH[sym]
        n1 = max(1, (n + 1) // 2)
        return phon(s, n1, nxt_vowel) + phon(e, n - n1, nxt_vowel)
    if sym == "HH":                       # both filters sit on the vowel, excited by noise
        return [[nf1, nf2, 5, 7, 11, 11] for _ in range(n)]
    if sym == "S":                        # last frame = quiet hand-over while X sweeps
        return [[250, S_HZ, 0, 9, 0, 11]] * (n - 1) + [[nf1, S_HZ, 4, 1, 0, 11]]
    if sym == "Z":
        return [[250, S_HZ, 5, 7, 0, 11]] * (n - 1) + [[250, S_HZ, 4, 1, 0, 11]]
    if sym == "V":                        # voice bar + weak half-noise at labial locus
        return [[250, 900, 6, 4, 0, 6] for _ in range(n)]
    if sym == "K":                        # closure + compact mid-frequency burst
        loc = min(2400.0, max(1700.0, nf2 * 1.25 + 300))
        return [[nf1, loc, 0, 0, 0, 11]] * (n - 1) + [[nf1, loc, 2, 10, 11, 11]]
    if sym in ("T", "P"):                 # closure, aspirated burst, hand-over to the vowel
        loc, v2 = {"T": (4100.0, 9), "P": (800.0, 5)}[sym]
        fr = [[nf1, loc, 0, 0, 0, 11]] * (n - 2) + [[nf1, loc, 2, v2, 11, 11]]
        return fr + [[nf1, loc, 5, 3, 0, 11]]
    if sym == "D":                        # voice bar, then half-noise release above the vowel F2
        return [[200, 2900.0, 2, 0, 0, 0]] * (n - 1) + [[250, 2900.0, 6, 6, 0, 6]]
    if sym == "TF":                       # word-final t: closure + weak release noise
        return [[300, 3600.0, 0, 0, 0, 11]] * (n - 1) + [[300, 3600.0, 0, 5, 0, 11]]
    if sym == "DF":                       # word-final d: voice bar + tiny schwa release
        return [[200, 1800.0, 3, 0, 0, 0]] * (n - 1) + [[400, 1800.0, 4, 2, 0, 0]]
    if sym == "GF":                       # word-final g: voice bar + voiced velar burst
        return [[200, 1800.0, 3, 0, 0, 0]] * (n - 1) + [[350, 1800.0, 4, 5, 0, 8]]
    if sym == "_":
        return [None] * n
    raise KeyError(sym)


# The sentence.  (phoneme, steps, note)   note = MIDI number, 45 = A2 = 110 Hz
TEXT = "Hello. I am Claude. I am an A.I. Now I live in a KORG D S ten. " \
       "I only have two voices. Nice to meet you."
SCRIPT = [
    ("_", 2, 47),
    # Hello.
    ("HH", 1, 47), ("AX", 2, 47), ("L", 2, 48), ("OW", 5, 50), ("_", 3, 45),
    # I am Claude.
    ("AY", 4, 48), ("AE", 2, 47), ("M", 2, 46), ("K", 2, 49), ("L", 2, 49),
    ("AO", 5, 48), ("DF", 2, 43), ("_", 3, 43),
    # I am an A.I.
    ("AY", 3, 47), ("AE", 2, 47), ("M", 2, 46), ("AX", 1, 46), ("N", 2, 47),
    ("EY", 4, 50), ("AY", 5, 47), ("_", 3, 43),
    # Now I live in a KORG D S ten.
    ("N", 2, 47), ("AW", 4, 49), ("AY", 3, 47), ("L", 2, 47), ("IH", 2, 48),
    ("V", 1, 47), ("IH", 2, 47), ("N", 2, 46), ("AX", 2, 46),
    ("K", 2, 49), ("AO", 3, 49), ("R", 3, 48), ("GF", 2, 46), ("_", 2, 46),
    ("D", 2, 48), ("IY", 4, 50), ("_", 2, 48),
    ("EH", 3, 49), ("S", 3, 48), ("_", 2, 48),
    ("T", 3, 48), ("EH", 3, 47), ("N", 4, 43), ("_", 3, 43),
    # I only have two voices.
    ("AY", 3, 47), ("OW", 3, 50), ("N", 2, 48), ("L", 1, 47), ("IY", 2, 47),
    ("HH", 1, 47), ("AE", 3, 48), ("V", 1, 46),
    ("T", 3, 50), ("UW", 3, 50), ("V", 2, 47), ("OY", 4, 48), ("S", 3, 46),
    ("IH", 2, 44), ("Z", 2, 43), ("_", 3, 43),
    # Nice to meet you.
    ("N", 2, 47), ("AY", 4, 50), ("S", 3, 48), ("T", 3, 47), ("AX", 1, 46),
    ("M", 2, 46), ("IY", 3, 48), ("TF", 2, 46), ("Y", 1, 45), ("UW", 5, 43),
    ("_", 3, 41),
]
JUMP_MASK = 3      # SYN2 X jump (levels/step) next to a noise step that gets its volume ducked


def first_vowel(i):
    for sym, _, _ in SCRIPT[i + 1:]:
        if sym in DIPH:
            sym = DIPH[sym][0]
        if sym in V and sym not in ("M", "N"):
            return V[sym][0], V[sym][1]
        if sym == "_":
            break
    return 500, 1400


def compile_script():
    frames, notes = [], []
    for i, (sym, n, note) in enumerate(SCRIPT):
        fr = phon(sym, n, first_vowel(i))
        frames += fr
        notes += [note] * len(fr)
    N = len(frames)
    N16 = -(-N // 16) * 16                               # pad to whole patterns
    frames += [None] * (N16 - N)
    notes += [notes[-1]] * (N16 - N)
    s1, s2 = [], []
    for i, fr in enumerate(frames):
        if fr is None:                                   # rest: park X on the next sound
            j = next((k for k in range(i, N16) if frames[k] is not None), None)
            f1, f2 = (frames[j][0], frames[j][1]) if j is not None else (500, 1400)
            s1.append(dict(note=notes[i], gate=0, vol=0, pan=2, x=q_hz(GRID1, f1), y=0))
            s2.append(dict(note=notes[i], gate=0, vol=0, pan=2, x=q_hz(GRID2, f2), y=0))
        else:
            f1, f2, a, b, y1, y2 = fr
            s1.append(dict(note=notes[i], gate="L", vol=a, pan=2, x=q_hz(GRID1, f1), y=y1))
            s2.append(dict(note=notes[i], gate="L", vol=b, pan=2, x=q_hz(GRID2, f2), y=y2))
    # authoring rule: hide big SYN2 sweeps into/out of noise (they chirp) behind the VOLUME lane
    for i in range(N16 - 1):
        noisy = s2[i]["y"] > 0 or s2[i + 1]["y"] > 0
        if s2[i]["gate"] and noisy and abs(s2[i + 1]["x"] - s2[i]["x"]) > JUMP_MASK:
            s2[i]["vol"] = min(s2[i]["vol"], 1)
    return s1, s2


def to_patterns(steps):
    return [steps[i:i + 16] for i in range(0, len(steps), 16)]


def render(wav="ds10_talk.wav", js="ds10_talk_sequence.json"):
    s1, s2 = compile_script()
    a = Synth("SYN1", SYN1_PATCH).render(s1, BPM, seed=1)
    b = Synth("SYN2", SYN2_PATCH).render(s2, BPM, seed=2)
    mix = master_fx(a + b, FX, BPM)
    mix = mix / np.max(np.abs(mix)) * 0.89               # mixer master level
    wavfile.write(wav, FS, (mix * 32767).astype(np.int16))
    data = {
        "text": TEXT, "bpm": BPM, "step": "16th note = %.1f ms" % (60000.0 / BPM / 4),
        "fx": FX, "SYN1": SYN1_PATCH, "SYN2": SYN2_PATCH,
        "kaoss_x_grid_hz": {"SYN1": [round(v) for v in GRID1], "SYN2": [round(v) for v in GRID2]},
        "patterns": {"SYN1": to_patterns(s1), "SYN2": to_patterns(s2)},
        "script": SCRIPT,
    }
    with open(js, "w") as f:
        json.dump(data, f, indent=1)
    print("steps: %d (%d patterns), %.2f s -> %s" % (len(s1), len(s1) // 16, len(mix) / FS, wav))
    return mix, s1, s2, a, b


if __name__ == "__main__":
    render()
