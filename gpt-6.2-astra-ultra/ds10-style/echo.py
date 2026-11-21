#!/usr/bin/env python3
"""Render Echo offline: python echo.py [score.json] [-o echo.wav].
Requires Python 3.10+ and NumPy. The first two fields in a row are comments.
Only the six sequencer lanes on each of two mono patches feed the synthesizer.
"""
import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WAVES = {"triangle", "saw", "square", "white_noise"}
GATES = {"25", "50", "75", "100", "LEG"}
PATCH_FIELDS = {"name", "vco1", "vco2", "vco2_pitch_semitones", "sync", "vcf_mode",
                "cutoff_hz", "peak_q", "eg", "vca_mode", "vca_level", "drive",
                "portamento_ms", "x", "y", "kaoss", "patch_cables", "mg", "fx"}


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def extract_tracks(project):
    patches, bank = project["patch"], project["patterns"]
    require(10 <= project["bpm"] <= 250 and len(patches) == 2, "BPM or track count")
    require(set(project["model"]) == {"eg_int_max_octaves"}, "Unknown model controls")
    require(project["model"]["eg_int_max_octaves"] == 4, "Model modulation range is fixed")
    require(1 <= len(bank) <= 16 and 1 <= len(project["song"]) <= 100, "Song/bank size")
    for p in patches:
        require(set(p) == PATCH_FIELDS, "Unexpected patch fields")
        require(p["vco1"] in WAVES and p["vco2"] in WAVES, "VCO wave")
        require(p["vcf_mode"] in ("LPF","BPF","HPF"), "VCF type")
        require(p["vca_mode"] in ("EG","GATE"), "VCA mode")
        require(p["x"] in ("VCF EG INT","VCF CUTOFF") and p["y"] == "VCO BALANCE", "XY assignment")
        require(p["kaoss"] in ("SMOOTH","HOLD"), "KAOSS mode")
        require(p["mg"] == "unpatched" and p["fx"] == "off" and p["patch_cables"] == [],
                "This renderer implements only the unpatched, dry configuration")
        require(set(p["eg"]) == {"attack_ms","decay_ms","sustain","release_ms"}, "ADSR")
        require(0 <= p["eg"]["sustain"] <= 1 and 0 < p["peak_q"] <= 30, "Sustain or Q")
        require(min(p["eg"][k] for k in ("attack_ms","decay_ms","release_ms")) >= 0, "EG time")
        require(0 <= p["vca_level"] <= 1 and 0 <= p["drive"] <= 1, "VCA level/drive")
        require(0 < p["cutoff_hz"] <= 16000 and p["portamento_ms"] >= 0, "Cutoff or porta")
    for name, pattern in bank.items():
        require(1 <= len(pattern) <= 16, f"{name}: pattern length")
        for i, (word_comment, gesture_comment, a, b) in enumerate(pattern):
            for lane in (a, b):
                note, gate, volume, pan, x, y = lane
                require(note is None or type(note) is int and 12 <= note <= 107, "C0..B7")
                require(gate in GATES, "Gate")
                require(all(type(v) is int and 0 <= v <= 11 for v in (volume, x, y)), "Volume or XY")
                require(type(pan) is int and -2 <= pan <= 2, "Pan")
    rows = [r for name in project["song"] for r in bank[name]]
    return [[r[j] for r in rows] for j in (2, 3)]


def controls(lanes, idx, frac, column, mode):
    values = np.array([row[column] for row in lanes], dtype=float)
    curr = values[idx]
    if mode == "HOLD":
        return curr
    after = values[np.minimum(idx + 1, len(values) - 1)]
    return curr + (after - curr) * frac


def amplitude_envelope(lanes, bounds, fs, eg):
    """One ADSR per voice. LEG ties into the following note without retrigger."""
    adsr = np.zeros(bounds[-1])
    gate_signal = np.zeros(bounds[-1])
    state, current, count, increment = "idle", 0., 0, 0.
    frames = {k: round(eg[k + "_ms"]*fs/1000) for k in ("attack","decay","release")}
    targets = {"attack": 1., "decay": eg["sustain"], "release": 0.}
    successor = {"attack": "decay", "decay": "sustain", "release": "idle"}

    def start(stage):
        nonlocal state, current, count, increment
        state = stage
        while state in frames:
            count = frames[state]
            if count:
                increment = (targets[state] - current) / count
                break
            current = targets[state]
            state = successor[state]

    def write(pos, end):
        nonlocal current, count
        while pos < end:
            if state not in frames:
                adsr[pos:end] = current
                return
            n = min(count, end-pos)
            adsr[pos:pos+n] = current + np.arange(1, n+1) * increment
            current = float(adsr[pos+n-1])
            pos += n
            count -= n
            if count == 0:
                current = targets[state]
                start(successor[state])

    previous_tie = False
    for i, (note, g, *_) in enumerate(lanes):
        a, b = bounds[i:i+2]
        if note is None:
            if previous_tie:
                start("release")
            write(a, b)
            previous_tie = False
            continue
        if not previous_tie:
            start("attack")
        stop = b if g == "LEG" else a + round((b-a)*int(g)/100)
        gate_signal[a:stop] = 1.
        write(a, stop)
        previous_tie = g == "LEG"
        if not previous_tie:
            start("release")
            write(stop, b)
    return adsr, gate_signal


def pitch_track(lanes, bounds, fs, porta_ms):
    remembered = next((r[0] for r in lanes if r[0] is not None), 48)
    pitch = float(remembered)
    result = np.empty(bounds[-1])
    for row, a, b in zip(lanes, bounds, bounds[1:]):
        if row[0] is not None:
            remembered = row[0]
        if porta_ms > 0 and pitch != remembered:
            decay = np.exp(-np.arange(1,b-a+1) / (fs*porta_ms/1000))
            result[a:b] = remembered + (pitch-remembered)*decay
            pitch = result[b-1]
        else:
            result[a:b] = pitch = remembered
    return 440. * np.exp2((result-69.)/12.)


def oscillator(shape, phase, rng):
    if shape == "saw":
        return 2*phase - 1
    if shape == "square":
        return np.where(phase < .5, 1., -1.)
    if shape == "triangle":
        return 1 - 4*np.abs(phase-.5)
    return rng.uniform(-1., 1., len(phase))


def vcf(signal, frequency, q, mode, fs):
    """A two-integrator, topology-preserving state-variable VCF. One output tap."""
    g = np.tan(np.pi*np.clip(frequency, 15., fs*.45)/fs)
    k = 1/q
    a = 1/(1+g*(g+k))
    b = g*a
    c = g*b
    tap = {"LPF":0, "BPF":1, "HPF":2}[mode]
    out = np.empty_like(signal)
    s1 = s2 = 0.
    for i,(x,ai,bi,ci) in enumerate(zip(signal,a,b,c)):
        v = x-s2
        bp = ai*s1 + bi*v
        lp = s2 + bi*s1 + ci*v
        s1, s2 = 2*bp-s1, 2*lp-s2
        out[i] = lp if tap == 0 else (bp if tap == 1 else x-k*bp-lp)
    return out


def voice(patch, lanes, fs, bounds, idx, frac, seed, eg_range):
    rng = np.random.default_rng(seed)
    eg, gate = amplitude_envelope(lanes,bounds,fs,patch["eg"])
    hz = pitch_track(lanes,bounds,fs,patch["portamento_ms"])
    cycles = np.cumsum(hz/fs)
    master_phase = cycles % 1.
    ratio = 2**(patch["vco2_pitch_semitones"]/12.)
    slave_phase = (master_phase*ratio if patch["sync"] else cycles*ratio) % 1.
    o1 = oscillator(patch["vco1"], master_phase, rng)
    o2 = oscillator(patch["vco2"], slave_phase, rng)
    x = controls(lanes,idx,frac,4,patch["kaoss"])/11.
    mix = controls(lanes,idx,frac,5,patch["kaoss"])/11.
    source = (1-mix)*o1 + mix*o2
    if patch["x"] == "VCF EG INT":
        cutoff = patch["cutoff_hz"] * np.exp2((2*x-1)*eg_range*eg)
    else:
        # Direct X-to-cutoff comparison patch: neutral VCF EG intensity.
        cutoff = 30.*(16000./30.)**x
    sig = vcf(source, cutoff, patch["peak_q"], patch["vcf_mode"], fs)
    if patch["drive"]:
        sig = np.tanh((1+4*patch["drive"])*sig)
    sig *= eg if patch["vca_mode"] == "EG" else gate
    sig *= patch["vca_level"] * controls(lanes,idx,frac,2,"HOLD")/11.
    pan = controls(lanes,idx,frac,3,"HOLD")
    angle = (pan+2)*np.pi/8.
    return np.column_stack((sig*np.cos(angle), sig*np.sin(angle)))


def render(project):
    lanes = extract_tracks(project)
    fs = project["sample_rate"]
    bounds = np.rint(np.arange(len(lanes[0])+1)*(fs*15./project["bpm"])).astype(int)
    lengths = np.diff(bounds)
    idx = np.repeat(np.arange(len(lengths)), lengths)
    frac = (np.arange(bounds[-1])-bounds[idx])/lengths[idx]
    stereo = np.zeros((bounds[-1],2))
    for i, (patch, track) in enumerate(zip(project["patch"],lanes)):
        stereo += voice(patch,track,fs,bounds,idx,frac,project["noise_seed"]+i,
                        project["model"]["eg_int_max_octaves"])
    return stereo * project["master_level"]


def write_wave(output, project, stereo):
    peak = np.max(np.abs(stereo))
    require(np.isfinite(stereo).all() and peak < 1, f"Master clips or contains NaNs: peak={peak}")
    pcm = np.rint(stereo*32767).astype("<i2")
    with wave.open(str(output), "wb") as stream:
        stream.setparams((2,2,project["sample_rate"],0,"NONE","not compressed"))
        stream.writeframes(pcm.tobytes())
    print(f"{output}: {len(pcm)/project['sample_rate']:.3f} s, "
          f"{project['sample_rate']} Hz, stereo PCM16, peak {20*math.log10(peak):.2f} dBFS")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("score", nargs="?", type=Path, default=HERE/"echo_score.json")
    cli.add_argument("-o", "--output", type=Path, default=Path("echo.wav"))
    args = cli.parse_args()
    project = json.loads(args.score.read_text(encoding="utf-8"))
    write_wave(args.output,project,render(project))
