"""Reproduce controls-only playback and diagnostic measurements. No audio is written.
Usage: python check.py [path/to/DS10_voice.wav]
NumPy's FFT is used here solely to measure an existing waveform. echo.py never imports this file.
"""
import argparse
import copy
import json
import wave
from pathlib import Path

import numpy as np

import echo
from score_source import make_project


def read_pcm(path, sample_rate):
    with wave.open(str(path), "rb") as wav:
        assert (wav.getsampwidth(), wav.getframerate(), wav.getnchannels()) == (2, sample_rate, 2)
        return np.frombuffer(wav.readframes(wav.getnframes()), "<i2").reshape(-1, 2)


def measurements(audio, rows, project):
    fs = project["sample_rate"]
    step = fs * 15 / project["bpm"]
    peaks = []
    centers = {"s": [], "sh": [], "z": []}
    for i, row in enumerate(rows):
        phone = row[1]
        repeated_i = phone == "ii" and i+1 < len(rows) and rows[i+1][1] == "ii"
        if phone not in centers and not repeated_i:
            continue
        # Use the same middle segment and analysis bands in every condition.
        section = audio[round((i+.15)*step):round((i+.70)*step), 0]
        power = np.abs(np.fft.rfft(np.hanning(len(section))*section, n=8192))**2
        f = np.fft.rfftfreq(8192, 1/fs)
        if repeated_i:
            band = (f >= 1600) & (f <= 3500)
            peaks.append(f[band][power[band].argmax()])
        if phone in centers:
            band = (f >= 1800) & (f <= 11000)
            centers[phone].append(np.average(f[band], weights=power[band]))
    return len(peaks), np.median(peaks), {p: np.median(c) for p, c in centers.items()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("wav", type=Path, nargs="?")
    args = cli.parse_args()
    project = json.loads((echo.HERE/"echo_score.json").read_text(encoding="utf-8"))
    assert project == make_project()
    rows = [row for name in project["song"] for row in project["patterns"][name]]
    audio = echo.render(project)
    assert (len(audio), audio.shape[1]) == (441000, 2)
    assert np.isfinite(audio).all() and np.max(np.abs(audio)) < 1
    if args.wav:
        assert np.array_equal(np.rint(audio*32767).astype("<i2"), read_pcm(args.wav, project["sample_rate"]))
        print("Supplied WAV exactly matches the regenerated PCM samples.")
    blank = copy.deepcopy(project)
    blank["text"] = ""
    for pattern in blank["patterns"].values():
        for row in pattern:
            row[:2] = ["", ""]
    assert np.array_equal(echo.render(blank), audio)
    print(f"Labels removed: identical signal. {len(rows)} steps, {len(project['patterns'])} patterns.")
    smooth = copy.deepcopy(project)
    smooth["patch"][1]["kaoss"] = "SMOOTH"
    for name, signal in (("Direct cutoff", echo.render(make_project(direct=True))),
                         ("Both smooth", echo.render(smooth)), ("Final", audio)):
        count, peak, centers = measurements(signal, rows, project)
        print(f"{name} | /i/ upper peak (median of {count}) {peak:.0f} Hz | "
              + " | ".join(f"{k} centroid {v:.0f} Hz" for k,v in centers.items()))


if __name__ == "__main__":
    main()
