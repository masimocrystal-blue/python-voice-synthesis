#!/usr/bin/env python3
"""Objective checks for formant_tts output (analysis only; not used in synthesis).

1. LPC formant measurement at each vowel midpoint vs. the intended targets.
2. Level of each consonant noise relative to vowels.
3. Optional: run PocketSphinx (a speech recogniser) on the result.
"""
import sys
import numpy as np
from scipy import signal
from scipy.linalg import solve_toeplitz
import formant_tts as T


def lpc_formants(x, order=24):
    x = signal.lfilter([1, -0.97], [1], x) * np.hamming(len(x))
    r = np.correlate(x, x, 'full')[len(x) - 1:len(x) + order]
    a = np.concatenate([[1], -solve_toeplitz(r[:-1], r[1:])])
    rts = [z for z in np.roots(a) if z.imag > 0]
    out = []
    for z in rts:
        f = np.angle(z) * T.FS / (2 * np.pi)
        bw = -np.log(abs(z)) * T.FS / np.pi
        if 150 < f < 4000 and bw < 500:
            out.append(f)
    return sorted(out)


def main(text, asr=True):
    y, segs, tr = T.synthesize(text)
    print(f'duration {len(y)/T.FS:.2f}s  peak {abs(y).max():.2f}  '
          f'rms {np.sqrt(np.mean(y**2)):.3f}  nan={np.isnan(y).any()}')
    print(f'F0 range {tr["f0"].min():.0f}-{tr["f0"].max():.0f} Hz')
    errs = []
    print('vowel  t(s)  target F1/F2/F3 -> measured')
    for s in segs:
        if not s['vowel'] or s['dur'] < 0.05:
            continue
        c = int((s['t0'] + 0.5 * s['dur']) * T.FS)
        w = int(0.02 * T.FS)
        tgt = [tr['F'][k][c] for k in range(3)]
        m = lpc_formants(y[c - w:c + w])
        near = [min(m, key=lambda f: abs(f - t)) if m else np.nan for t in tgt]
        e = [abs(a - b) / b * 100 for a, b in zip(near, tgt)]
        errs.append(e)
        print(f'{s["label"]:4s} {s["t0"]:5.2f}  '
              + '/'.join(f'{v:4.0f}' for v in tgt) + ' -> '
              + '/'.join(f'{v:4.0f}' for v in near)
              + ('   <-- off' if max(e[:2]) > 12 else ''))
    errs = np.array(errs)
    print('mean abs error %: F1 {:.1f}  F2 {:.1f}  F3 {:.1f}'.format(*np.nanmean(errs, 0)))

    vr = np.sqrt(np.mean(y[tr['AV'] > 0.8] ** 2))
    print('consonant level re vowel RMS (dB):')
    lv = {}
    for s in segs:
        if s['noise'] or s['nasal'] or s['label'] == 'HH':
            a, b = int(s['t0'] * T.FS), int(s['t1'] * T.FS)
            lv.setdefault(s['label'], []).append(
                20 * np.log10(np.sqrt(np.mean(y[a:b] ** 2)) / vr + 1e-9))
    print('  ' + '  '.join(f'{k}:{np.mean(v):.0f}' for k, v in sorted(lv.items())))

    if asr:
        try:
            from pocketsphinx import Decoder
        except ImportError:
            print('pocketsphinx not installed; skipping ASR')
            return
        pcm = signal.resample_poly(y, 16000, T.FS)
        pcm = np.round(pcm * 32767 * 0.9).astype(np.int16).tobytes()
        words = sorted({w.lower() for w in __import__('re').findall(r"[A-Za-z']+", text)})
        for name, kw in (('free dictation (large vocabulary)', {}),):
            d = Decoder(samprate=16000, loglevel='FATAL', **kw)
            d.start_utt(); d.process_raw(pcm, full_utt=True); d.end_utt()
            print(f'ASR {name}:\n   ', d.hyp().hypstr if d.hyp() else '(nothing)')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else T.DEFAULT_TEXT)
