#!/usr/bin/env python3
"""
formant_tts.py - a rule-based formant speech synthesizer written from scratch.

No TTS engine, no trained model, no recorded audio. Every sample is computed
from numbers in this file: a glottal pulse model, time-varying resonators,
filtered noise, and hand-written rules for duration and pitch.

Dependencies: NumPy, SciPy (lfilter / resample_poly / gaussian smoothing /
WAV writer), and optionally Matplotlib for the spectrogram picture.

Usage:
    python formant_tts.py                       # the default self-introduction
    python formant_tts.py "This is my voice." -o out.wav --spectrogram out.png
    python formant_tts.py "Hello {W ER1 L D}."  # ARPABET in braces for any word
"""
import argparse
import re

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d

FS = 22050          # output sample rate (Hz)
OQ = 0.60           # glottal open quotient
F_HIGH = [(3300, 250), (3850, 300), (4900, 500),    # fixed F4-F9 (freq, bandwidth)
          (5900, 700), (6900, 900), (7900, 1000)]
FNP = 270           # nasal pole frequency
TILT_HZ = 4000      # source spectral tilt corner
HF_EMPH = 0.3       # high-frequency emphasis on the voiced branch (0..1)
BURST_GAIN = 1.0    # stop burst level multiplier
ASP_GAIN = 0.40     # aspiration noise level
ASP_LP = 6000       # aspiration low-pass corner (Hz)
VELAR_BACK = 1250   # F2 locus of K/G next to back vowels
VELAR_BURST = 100   # K/G burst centre = locus + this

DEFAULT_TEXT = ("Hello. I am Claude, an AI made by Anthropic. "
                "This voice is not a recording. "
                "Every sample was computed from scratch in Python.")

# ----------------------------------------------------------------------------
# 1. Phoneme table (ARPABET).  Formant targets are for an adult male voice.
#    kind: V vowel, G glide/liquid, N nasal, F fricative, S stop, A affricate, H
# ----------------------------------------------------------------------------
PH = {
    # monophthongs      F1    F2    F3
    'IY': dict(kind='V', f=(270, 2290, 3010), dur=105),
    'IH': dict(kind='V', f=(390, 1990, 2550), dur=70),
    'EH': dict(kind='V', f=(530, 1840, 2480), dur=85),
    'AE': dict(kind='V', f=(660, 1720, 2410), dur=125),
    'AA': dict(kind='V', f=(730, 1090, 2440), dur=125),
    'AO': dict(kind='V', f=(570, 840, 2410), dur=125),
    'AH': dict(kind='V', f=(640, 1190, 2390), dur=80),
    'AX': dict(kind='V', f=(500, 1450, 2450), dur=55),      # schwa (= AH0)
    'UH': dict(kind='V', f=(440, 1020, 2240), dur=75),
    'UW': dict(kind='V', f=(300, 870, 2240), dur=110),
    'ER': dict(kind='V', f=(490, 1350, 1690), dur=110),
    # diphthongs: start target f -> end target f2
    'EY': dict(kind='V', f=(480, 1900, 2520), f2=(330, 2200, 2700), dur=150),
    'AY': dict(kind='V', f=(730, 1200, 2500), f2=(400, 1950, 2600), dur=165),
    'OW': dict(kind='V', f=(540, 1050, 2400), f2=(400, 850, 2300), dur=150),
    'AW': dict(kind='V', f=(730, 1250, 2450), f2=(450, 900, 2300), dur=165),
    'OY': dict(kind='V', f=(550, 850, 2450), f2=(380, 1950, 2550), dur=180),
    # glides / liquids
    'W': dict(kind='G', f=(290, 610, 2150), dur=65),
    'Y': dict(kind='G', f=(260, 2200, 3000), dur=60),
    'R': dict(kind='G', f=(330, 1060, 1500), dur=70),
    'L': dict(kind='G', f=(360, 1000, 2700), dur=65),
    # nasals (murmur: low F1, heavily damped upper formants, nasal zero)
    'M': dict(kind='N', f=(280, 1000, 2300), dur=70),
    'N': dict(kind='N', f=(280, 1650, 2700), dur=60),
    'NG': dict(kind='N', f=(280, 2000, 2600), dur=75, velar=True),
    # fricatives: noise = name of a spectrum in NOISE; amp is relative to the
    # RMS level of a full vowel
    'S': dict(kind='F', f=(280, 1700, 2600), dur=105, amp=0.33,
              noise='S'),
    'SH': dict(kind='F', f=(260, 1900, 2500), dur=110, amp=0.38,
               noise='SH'),
    'F': dict(kind='F', f=(280, 1000, 2300), dur=95, amp=0.10,
              noise='F'),
    'TH': dict(kind='F', f=(280, 1400, 2550), dur=90, amp=0.085,
               noise='TH'),
    'Z': dict(kind='F', f=(260, 1700, 2600), dur=75, amp=0.17, voiced=True,
              noise='S'),
    'ZH': dict(kind='F', f=(260, 1900, 2500), dur=70, amp=0.19, voiced=True,
               noise='SH'),
    'V': dict(kind='F', f=(260, 1000, 2300), dur=60, amp=0.05, voiced=True,
              noise='F'),
    'DH': dict(kind='F', f=(260, 1400, 2550), dur=40, amp=0.035, voiced=True,
               noise='TH'),
    'HH': dict(kind='H', f=(500, 1500, 2500), dur=60),
    # stops: closure ms, burst ms, burst noise, aspiration (VOT) ms
    'P': dict(kind='S', f=(220, 900, 2300), clo=70, burst=6, vot=50, amp=0.16,
              noise='P'),
    'T': dict(kind='S', f=(220, 1700, 2700), clo=55, burst=12, vot=55, amp=0.36,
              noise='T'),
    'K': dict(kind='S', f=(220, 2000, 2500), clo=60, burst=18, vot=60, amp=0.30,
              noise='K', velar=True),
    'B': dict(kind='S', f=(220, 900, 2300), clo=60, burst=5, vot=6, amp=0.10,
              voiced=True, noise='P'),
    'D': dict(kind='S', f=(220, 1700, 2700), clo=50, burst=8, vot=8, amp=0.20,
              voiced=True, noise='T'),
    'G': dict(kind='S', f=(220, 2000, 2500), clo=55, burst=12, vot=10, amp=0.18,
              voiced=True, noise='K',
              velar=True),
    # affricates = stop closure + fricative release
    'CH': dict(kind='A', f=(260, 1900, 2500), clo=50, fric='SH', fdur=80),
    'JH': dict(kind='A', f=(260, 1900, 2500), clo=40, fric='ZH', fdur=60,
               voiced=True),
}
# Noise spectra as (Hz, dB) break-points; an FIR filter is designed from each.
NOISE = {
    'S': [(0, -60), (2500, -45), (3500, -24), (4500, -6), (5500, 0),
          (8000, -3), (11025, -14)],
    'SH': [(0, -60), (1200, -40), (1800, -14), (2500, 0), (3500, -3),
           (5000, -13), (7000, -26), (11025, -40)],
    'F': [(0, -30), (800, -12), (2000, -6), (4000, -3), (8000, 0), (11025, -3)],
    'TH': [(0, -30), (1000, -12), (2000, -5), (5000, -2), (8000, 0), (11025, 0)],
    'P': [(0, -10), (500, 0), (1500, -6), (3000, -15), (6000, -25), (11025, -35)],
    'T': [(0, -40), (2000, -20), (3000, -6), (4000, 0), (6000, -2),
          (9000, -10), (11025, -16)],
}


def velar_spectrum(fk):
    """Compact burst spectrum for K/G, centred near the vowel's F2."""
    return [(0, -30), (fk - 600, -18), (fk - 150, 0), (fk + 150, 0),
            (fk + 700, -14), (fk + 2200, -12), (fk + 3500, -24), (11025, -38)]


VOICELESS = {'P', 'T', 'K', 'F', 'TH', 'S', 'SH', 'CH', 'HH'}

# ----------------------------------------------------------------------------
# 2. Pronunciation lexicon (ARPABET, stress digits on vowels).
#    Text input is limited to these words; anything else can be given as
#    phonemes in braces, e.g.  "Hello {W ER1 L D}".
# ----------------------------------------------------------------------------
LEXICON = {
    'hello': 'HH AH0 L OW1', 'i': 'AY1', 'am': 'AE1 M', 'claude': 'K L AO1 D',
    'an': 'AH0 N', 'a': 'AH0', 'ai': 'EY1 AY1', 'made': 'M EY1 D', 'by': 'B AY1',
    'anthropic': 'AE0 N TH R AA1 P IH0 K', 'this': 'DH IH1 S',
    'voice': 'V OY1 S', 'is': 'IH0 Z', 'not': 'N AA1 T',
    'recording': 'R IH0 K AO1 R D IH0 NG', 'every': 'EH1 V R IY0',
    'sample': 'S AE1 M P AH0 L', 'was': 'W AH0 Z',
    'computed': 'K AH0 M P Y UW1 T IH0 D', 'from': 'F R AH0 M',
    'scratch': 'S K R AE1 CH', 'in': 'IH0 N', 'python': 'P AY1 TH AA0 N',
    # extra vocabulary so other sentences can be tried
    'the': 'DH AH0', 'quick': 'K W IH1 K', 'brown': 'B R AW1 N',
    'fox': 'F AA1 K S', 'jumps': 'JH AH1 M P S', 'over': 'OW1 V ER0',
    'lazy': 'L EY1 Z IY0', 'dog': 'D AO1 G', 'my': 'M AY1', 'name': 'N EY1 M',
    'you': 'Y UW1', 'are': 'AA1 R', 'we': 'W IY1', 'she': 'SH IY1',
    'he': 'HH IY1', 'it': 'IH1 T', 'and': 'AH0 N D', 'of': 'AH0 V',
    'to': 'T UW0', 'for': 'F AO1 R', 'with': 'W IH0 DH', 'that': 'DH AE1 T',
    'can': 'K AE1 N', 'speak': 'S P IY1 K', 'world': 'W ER1 L D',
    'good': 'G UH1 D', 'morning': 'M AO1 R N IH0 NG', 'thank': 'TH AE1 NG K',
    'yes': 'Y EH1 S', 'no': 'N OW1', 'one': 'W AH1 N', 'two': 'T UW1',
    'three': 'TH R IY1', 'four': 'F AO1 R', 'five': 'F AY1 V',
    'math': 'M AE1 TH', 'pure': 'P Y UH1 R', 'sound': 'S AW1 N D',
    'hear': 'HH IY1 R', 'machine': 'M AH0 SH IY1 N', 'vision': 'V IH1 ZH AH0 N',
    'today': 'T AH0 D EY1', 'very': 'V EH1 R IY0', 'nice': 'N AY1 S',
    'meet': 'M IY1 T', 'robot': 'R OW1 B AA0 T', 'sells': 'S EH1 L Z',
    'sea': 'S IY1', 'shells': 'SH EH1 L Z', 'shore': 'SH AO1 R', 'on': 'AA1 N',
}
PAUSE = {'.': 0.32, '!': 0.32, '?': 0.32, ',': 0.17, ';': 0.22, ':': 0.22}


def text_to_items(text):
    """Text -> flat list of phoneme items and pause items."""
    items = []

    def add_word(phones, word):
        for k, p in enumerate(phones):
            m = re.fullmatch(r'([A-Z]+)([012])?', p)
            if not m or m.group(1) not in PH:
                raise ValueError(f'unknown phoneme {p!r} in {word!r}')
            ph, st = m.group(1), m.group(2)
            st = int(st) if st is not None else None
            if ph == 'AH' and st == 0:
                ph = 'AX'
            items.append(dict(type='ph', ph=ph, stress=st, word=word,
                              wstart=(k == 0), wend=(k == len(phones) - 1)))

    for tok in re.findall(r"\{[^}]*\}|[A-Za-z']+|[.,;:!?]", text):
        if tok in PAUSE:
            items.append(dict(type='pause', dur=PAUSE[tok], kind=tok))
        elif tok.startswith('{'):
            add_word(tok[1:-1].split(), tok)
        else:
            w = tok.lower()
            if w not in LEXICON:
                raise ValueError(
                    f'"{tok}" is not in the lexicon. Add it to LEXICON or write '
                    f'its phonemes in braces, e.g. {{HH AH0 L OW1}}.')
            add_word(LEXICON[w].split(), w)
    if not items or items[-1]['type'] != 'pause':
        items.append(dict(type='pause', dur=0.3, kind='.'))
    return items


# ----------------------------------------------------------------------------
# 3. Phonemes -> acoustic segments (durations, targets, noise events)
# ----------------------------------------------------------------------------
def build_segments(items, rate=1.0):
    n = len(items)
    is_ph = lambda j: 0 <= j < n and items[j]['type'] == 'ph'
    kind = lambda j: PH[items[j]['ph']]['kind'] if is_ph(j) else None

    # mark everything from the last vowel of each phrase as phrase-final
    start = 0
    for j in range(n):
        if items[j]['type'] == 'pause':
            vs = [k for k in range(start, j) if kind(k) == 'V']
            if vs:
                for k in range(vs[-1], j):
                    items[k]['final'] = True
            start = j + 1

    def near_sonorant_f(j):
        """Formants of the nearest vowel/glide (next one preferred)."""
        k = j + 1
        while is_ph(k):
            if kind(k) in 'VG':
                return PH[items[k]['ph']]['f']
            k += 1
        k = j - 1
        while is_ph(k):
            if kind(k) in 'VG':
                d = PH[items[k]['ph']]
                return d.get('f2', d['f'])
            k -= 1
        return (500, 1500, 2500)

    def locus(j):
        d = PH[items[j]['ph']]
        if d.get('velar'):                       # velars follow the vowel
            f2v = near_sonorant_f(j)[1]
            if f2v > 1600:
                return (d['f'][0], 2250, 2650)
            if f2v > 1150:
                return (d['f'][0], 1600, 2300)
            return (d['f'][0], VELAR_BACK, 2200)
        return d['f']

    def onset_stressed(j):
        k = j + 1
        while is_ph(k) and not items[k]['wstart']:
            if kind(k) == 'V':
                return items[k]['stress'] == 1
            k += 1
        return False

    segs = []

    def seg(label, dur, fa, fb=None, av=0.0, ah=0.0, anchors=(0.1, 0.9),
            bw=(70, 100, 160), nasal=False, noise=None, **kw):
        s = dict(label=label, dur=dur / 1000.0 / rate, fa=fa,
                 fb=fb if fb is not None else fa, av=av, ah=ah, anchors=anchors,
                 bw=bw, nasal=nasal, noise=noise, accent=False, vowel=False)
        s.update(kw)
        segs.append(s)
        return s

    for j, it in enumerate(items):
        if it['type'] == 'pause':
            seg('_', it['dur'] * 1000 * rate, None, pause=it['kind'])
            continue
        ph, d, k = it['ph'], PH[it['ph']], PH[it['ph']]['kind']
        final = it.get('final', False)
        nxt = items[j + 1]['ph'] if is_ph(j + 1) else None
        cluster = (kind(j - 1) not in (None, 'V')) or (kind(j + 1) not in (None, 'V'))
        scale = (1.4 if final else 1.0)

        if k == 'V':
            dur = d['dur'] * scale
            st = it['stress']
            dur *= {1: 1.1, 2: 0.95, 0: 1.0 if ph == 'AX' else 0.78, None: 1.0}[st]
            if nxt in VOICELESS and not items[j + 1]['wstart']:
                dur *= 0.85
            if 'f2' in d:
                seg(ph, dur, d['f'], d['f2'], av=1.0, anchors=(0.2, 0.85),
                    vowel=True, accent=(st == 1))
            else:
                a0 = min(0.35, 45.0 / dur)
                seg(ph, dur, d['f'], av=1.0 if st != 0 else 0.8,
                    anchors=(a0, 1 - a0), vowel=True, accent=(st == 1))
        elif k == 'G':
            seg(ph, d['dur'] * scale * (0.85 if cluster else 1.0), d['f'],
                av=0.75, anchors=(0.3, 0.7), bw=(70, 110, 170))
        elif k == 'N':
            seg(ph, d['dur'] * scale * (0.9 if cluster else 1.0), locus(j),
                av=0.6, bw=(70, 120, 180), nasal=True, anchors=(0.0, 1.0))
        elif k == 'H':
            f = near_sonorant_f(j)
            seg(ph, d['dur'], f, av=0.0, ah=0.3, anchors=(0.0, 1.0),
                bw=(200, 200, 250))
        elif k == 'F':
            dur = d['dur'] * scale * (0.85 if cluster else 1.0)
            voiced = d.get('voiced', False)
            seg(ph, dur, d['f'], av=0.45 if voiced else 0.0,
                bw=(90, 200, 300), anchors=(0.0, 1.0),
                noise=dict(spec=NOISE[d['noise']], amp=d['amp'], env='fric',
                           voiced=voiced))
        elif k in 'SA':
            voiced = d.get('voiced', False)
            loc = locus(j)
            clo = d['clo'] * (0.85 if cluster else 1.0)
            after_s = is_ph(j - 1) and items[j - 1]['ph'] == 'S' and not it['wstart']
            seg(ph + 'cl', clo, loc, av=0.13 if voiced else 0.0,
                bw=(100, 400, 600), anchors=(0.0, 1.0))
            if k == 'A':
                fd = PH[d['fric']]
                seg(ph, d['fdur'] * scale, d['f'], av=0.3 if voiced else 0.0,
                    bw=(90, 200, 300), anchors=(0.0, 1.0),
                    noise=dict(spec=NOISE[fd['noise']], amp=fd['amp'], env='affr',
                               voiced=voiced))
                continue
            # release: burst + aspiration
            spec = velar_spectrum(loc[1] + VELAR_BURST) if d.get('velar') else NOISE[d['noise']]
            amp = d['amp']
            if nxt is None:                       # before a pause: weak release
                vot, amp = (25, amp * 0.6) if not voiced else (d['vot'], amp * 0.6)
            elif kind(j + 1) in 'SAF':            # stop + obstruent: no aspiration
                vot, amp = 0, amp * 0.5
            elif voiced:
                vot = d['vot']
            elif after_s:
                vot = 12
            elif onset_stressed(j) or it['wstart']:
                vot = d['vot']
            else:
                vot = d['vot'] * 0.5
            nf = near_sonorant_f(j) if kind(j + 1) in ('V', 'G') else loc
            seg(ph, d['burst'] + vot, loc, nf, av=0.0,
                ah=0.0 if (voiced or vot < 15) else 0.45, anchors=(0.0, 1.0),
                bw=(200, 200, 250),
                noise=dict(spec=spec, amp=amp, env='burst', voiced=False,
                           blen=d['burst'] / 1000.0))
    return segs


# ----------------------------------------------------------------------------
# 4. Segments -> control tracks, one value per output sample
# ----------------------------------------------------------------------------
def build_tracks(segs, f0_base=112.0):
    t = 0.0
    for s in segs:
        s['t0'], s['t1'] = t, t + s['dur']
        t += s['dur']
    n = int(round(t * FS))
    ts = np.arange(n) / FS
    idx = lambda x: int(round(x * FS))

    # formants: straight lines between per-segment anchors, then smoothed
    ax, ay = [], []
    for s in segs:
        if s['fa'] is None:
            continue
        a0, a1 = s['anchors']
        ax += [s['t0'] + a0 * s['dur'], s['t0'] + a1 * s['dur']]
        ay += [s['fa'], s['fb']]
    ay = np.array(ay, float)
    F = [gaussian_filter1d(np.interp(ts, ax, ay[:, k]), 0.006 * FS) for k in range(3)]

    def step(key, default, sigma):
        x = np.full(n, default, float)
        for s in segs:
            x[idx(s['t0']):idx(s['t1'])] = key(s)
        return gaussian_filter1d(x, sigma * FS)

    AV = step(lambda s: s['av'], 0.0, 0.004)
    AH = step(lambda s: s['ah'], 0.0, 0.004)
    BW = [step(lambda s, k=k: s['bw'][k], (70, 100, 160)[k], 0.008) for k in range(3)]
    FNZ = step(lambda s: 480.0 if s['nasal'] else FNP, FNP, 0.012)

    # F0: declining baseline per phrase + accent bumps + boundary tones
    f0 = np.full(n, f0_base)
    start = 0
    n_phr = sum(1 for s in segs if s['label'] == '_')
    p = 0
    for j, s in enumerate(segs):
        if s['label'] != '_':
            continue
        phrase = segs[start:j]
        start = j + 1
        if not phrase:
            continue
        ta, tb = phrase[0]['t0'], phrase[-1]['t1']
        top = f0_base * (1.08 - 0.03 * min(p, 2))
        i0, i1 = idx(ta), idx(s['t1'])
        tt = ts[i0:i1]
        line = top + (f0_base * 0.84 - top) * np.clip((tt - ta) / max(tb - ta, 0.3), 0, 1)
        accents = [q for q in phrase if q['accent']]
        for m, q in enumerate(accents):
            c = q['t0'] + 0.4 * q['dur']
            h = 0.20 if m == 0 else (0.16 if m == len(accents) - 1 else 0.11)
            line += f0_base * h * np.exp(-0.5 * ((tt - c) / 0.075) ** 2)
        if s.get('pause') == ',':                       # continuation rise
            line += f0_base * 0.10 * np.clip((tt - (tb - 0.16)) / 0.16, 0, 1)
        else:                                           # final fall
            line -= f0_base * 0.13 * np.clip((tt - (tb - 0.28)) / 0.28, 0, 1)
        f0[i0:i1] = line
        p += 1
    f0 = gaussian_filter1d(f0, 0.018 * FS)
    return dict(n=n, F=F, BW=BW, AV=AV, AH=AH, FNZ=FNZ, f0=f0)


# ----------------------------------------------------------------------------
# 5. Signal generation
# ----------------------------------------------------------------------------
def glottal_source(f0, rng, oversample=4):
    """Glottal flow derivative (KLGLOTT88-style polynomial pulse).

    Flow in the open phase is x^2 - x^3, so its derivative 2x - 3x^2 rises
    slowly and snaps shut at x = 1; that abrupt closure excites the formants.
    Generated at 4x rate and decimated to keep aliasing down.
    """
    n = len(f0)
    ts = np.arange(n) / FS
    # slow pitch flutter + fast random jitter so the voice is not robot-steady
    flutter = 0.006 * (np.sin(2 * np.pi * 12.7 * ts) + np.sin(2 * np.pi * 7.1 * ts)
                       + np.sin(2 * np.pi * 4.7 * ts))
    jitter = gaussian_filter1d(rng.standard_normal(n), 0.004 * FS)
    jitter *= 0.006 / (jitter.std() + 1e-12)
    f = np.repeat(f0 * (1 + flutter + jitter), oversample)
    phase = np.cumsum(f / (FS * oversample)) % 1.0
    x = phase / OQ
    d = np.where(x < 1.0, 2 * x - 3 * x * x, 0.0)
    d = signal.resample_poly(d, 1, oversample)[:n]
    ph = phase[::oversample][:n]
    shimmer = gaussian_filter1d(rng.standard_normal(n), 0.01 * FS)
    d *= 1 + 0.04 * shimmer / (shimmer.std() + 1e-12)
    a = np.exp(-2 * np.pi * TILT_HZ / FS)              # gentle spectral tilt
    d = signal.lfilter([1 - a], [1, -a], d)
    return d, ph


def resonator(x, f, bw, anti=False):
    """Time-varying 2nd-order resonator (or anti-resonator), sample by sample.

    y[n] = A x[n] + B y[n-1] + C y[n-2], coefficients recomputed every sample
    from the formant frequency f[n] and bandwidth bw[n] (Klatt 1980).
    """
    n = len(x)
    f = np.broadcast_to(np.asarray(f, float), (n,))
    bw = np.broadcast_to(np.asarray(bw, float), (n,))
    C = -np.exp(-2 * np.pi * bw / FS)
    B = 2 * np.exp(-np.pi * bw / FS) * np.cos(2 * np.pi * f / FS)
    A = 1 - B - C
    y = np.empty(n)
    xl = x.tolist()
    if anti:
        A, B, C = 1 / A, -B / A, -C / A
        Al, Bl, Cl = A.tolist(), B.tolist(), C.tolist()
        x1 = x2 = 0.0
        for i in range(n):
            xi = xl[i]
            y[i] = Al[i] * xi + Bl[i] * x1 + Cl[i] * x2
            x2, x1 = x1, xi
    else:
        Al, Bl, Cl = A.tolist(), B.tolist(), C.tolist()
        y1 = y2 = 0.0
        for i in range(n):
            yi = Al[i] * xl[i] + Bl[i] * y1 + Cl[i] * y2
            y[i] = yi
            y2, y1 = y1, yi
    return y


def shaped_noise(n, spec, rng):
    """White noise shaped to a target spectrum (FIR by frequency sampling)."""
    fr = [p[0] for p in spec]
    g = [10 ** (p[1] / 20) for p in spec]
    h = signal.firwin2(127, fr, g, fs=FS)
    y = signal.lfilter(h, [1.0], rng.standard_normal(n + 127))[127:]
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-12)


def noise_envelope(n, kind, blen=0.0):
    t = np.arange(n) / FS
    if kind == 'burst':                    # sharp attack, fast decay
        e = np.minimum(t / 0.001, 1.0) * np.exp(-t / max(blen * 0.7, 0.003))
    elif kind == 'affr':                   # abrupt onset, slow decay
        e = np.minimum(t / 0.004, 1.0) * np.minimum((t[-1] - t) / 0.03 + 1e-3, 1.0)
    else:                                  # fricative: smooth rise and fall
        r = min(0.025, t[-1] / 3 + 1e-4)
        e = np.minimum(t / r, 1.0) * np.minimum((t[-1] - t) / r, 1.0)
        e = np.sin(0.5 * np.pi * np.clip(e, 0, 1)) ** 2
    return e


def synthesize(text=DEFAULT_TEXT, rate=0.9, f0_base=112.0, seed=1):
    """Text -> (float waveform in [-1, 1], list of segments with timings)."""
    rng = np.random.default_rng(seed)
    segs = build_segments(text_to_items(text), rate)
    tr = build_tracks(segs, f0_base)
    n = tr['n']

    # source: voiced pulses + aspiration noise (+ a little breath in voicing)
    glot, phase = glottal_source(tr['f0'], rng)
    opening = (phase < OQ).astype(float)
    asp = rng.standard_normal(n)                       # aspiration: tilted noise
    a = np.exp(-2 * np.pi * ASP_LP / FS)
    asp = signal.lfilter([1 - a], [1, -a], asp)
    asp *= ASP_GAIN / asp.std()
    src = tr['AV'] * (glot + 0.15 * asp * opening) + tr['AH'] * asp

    # vocal tract: nasal zero/pole pair, then F1..F3 (moving), F4..F9 (fixed)
    y = resonator(src, tr['FNZ'], 100.0, anti=True)
    y = resonator(y, FNP, 100.0)
    for k in range(3):
        y = resonator(y, tr['F'][k], tr['BW'][k])
    for fc, bw in F_HIGH:
        c = -np.exp(-2 * np.pi * bw / FS)
        b = 2 * np.exp(-np.pi * bw / FS) * np.cos(2 * np.pi * fc / FS)
        y = signal.lfilter([1 - b - c], [1, -b, -c], y)

    if HF_EMPH:
        y = signal.lfilter([1, -HF_EMPH], [1 - HF_EMPH], y)

    # frication and bursts are added in parallel, scaled to the vowel level
    loud = tr['AV'] > 0.8
    vrms = np.sqrt(np.mean(y[loud] ** 2)) if loud.any() else 1.0
    for s in segs:
        nz = s['noise']
        if not nz:
            continue
        i0 = int(round(s['t0'] * FS))
        m = int(round(s['dur'] * FS))
        if nz['env'] == 'burst':
            m = min(int(round((nz['blen'] * 4 + 0.01) * FS)), n - i0)
        m = min(m, n - i0)
        if m < 8:
            continue
        e = noise_envelope(m, nz['env'], nz.get('blen', 0.0))
        g = BURST_GAIN if nz['env'] == 'burst' else 1.0
        z = shaped_noise(m, nz['spec'], rng) * e * nz['amp'] * vrms * g
        if nz['voiced']:                    # voiced fricatives pulse with voicing
            z *= 0.55 + 0.45 * opening[i0:i0 + m]
        y[i0:i0 + m] += z

    b, a = signal.butter(2, 60 / (FS / 2), 'high')     # remove DC / rumble
    y = signal.lfilter(b, a, y)
    y *= 0.89 / np.max(np.abs(y))
    return y, segs, tr


# ----------------------------------------------------------------------------
# 6. Output helpers
# ----------------------------------------------------------------------------
def save_wav(path, y):
    wavfile.write(path, FS, np.round(y * 32767).astype(np.int16))


def save_spectrogram(path, y, segs, title=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dur = len(y) / FS
    rows = max(1, int(np.ceil(dur / 5.5)))
    span = dur / rows
    f, t, S = signal.spectrogram(y, FS, window='hamming', nperseg=220,
                                 noverlap=176, nfft=1024)
    S = 10 * np.log10(S + 1e-12)
    vmax = S.max()
    fig, axes = plt.subplots(rows, 1, figsize=(16, 3.6 * rows), squeeze=False)
    for r in range(rows):
        ax = axes[r, 0]
        ax.pcolormesh(t, f / 1000, S, vmin=vmax - 65, vmax=vmax, cmap='magma',
                      shading='auto', rasterized=True)
        ax.set_xlim(r * span, (r + 1) * span)
        ax.set_ylim(0, 8)
        ax.set_ylabel('Frequency (kHz)')
        for s in segs:
            if s['label'] == '_' or s['label'].endswith('cl'):
                continue
            c = 0.5 * (s['t0'] + s['t1'])
            if r * span <= c < (r + 1) * span:
                ax.text(c, 8.15, s['label'], ha='center', va='bottom', fontsize=7)
                ax.axvline(s['t0'], color='w', lw=0.3, alpha=0.35)
    axes[-1, 0].set_xlabel('Time (s)')
    if title:
        fig.suptitle(title, fontsize=10, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('text', nargs='?', default=DEFAULT_TEXT)
    ap.add_argument('-o', '--out', default='speech.wav')
    ap.add_argument('--spectrogram', default=None, metavar='PNG')
    ap.add_argument('--rate', type=float, default=0.9, help='speaking rate')
    ap.add_argument('--f0', type=float, default=112.0, help='base pitch in Hz')
    args = ap.parse_args()
    y, segs, _ = synthesize(args.text, args.rate, args.f0)
    save_wav(args.out, y)
    print(f'{args.out}: {len(y) / FS:.2f} s, {len(segs)} segments')
    if args.spectrogram:
        save_spectrogram(args.spectrogram, y, segs, args.text)
        print(args.spectrogram)


if __name__ == '__main__':
    main()
