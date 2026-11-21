"""Authoring notes. This script only writes sequencer data, never audio."""
from pathlib import Path
import argparse
import json
import math

HERE = Path(__file__).resolve().parent
# Each articulation specifies two target bands in Hz, source balances, and volumes.
# They are rounded to actual control positions when printing the piano roll.
phones = {
    # label: F1, F2, noise blend 1/2 (0..11), volume 1/2
    'ay_mid':(595,1630,0,0,11,8), 'ih':(385,2020,0,0,9,8), 'ii':(305,2350,0,0,8,10),
    'eh':(570,1830,0,0,10,9), 'ey':(485,2030,0,0,10,9),
    'ae':(705,1710,0,0,11,8), 'aa':(745,1180,0,0,11,8),
    'uh':(465,1430,0,0,8,7),
    'oo':(330,900,0,0,8,7), 'oh':(495,936,0,0,10,7),
    'uu':(307,830,0,0,8,6), 'ow':(435,890,0,0,10,7),
    'm':(220,950,0,0,5,1), 'n':(265,1660,0,0,5,2),
    'l':(340,1160,0,0,6,4), 'darkl':(340,890,0,0,6,3),
    'r':(365,1280,0,0,8,6), 'y':(267,2330,0,0,7,9),
    'h':(490,1620,11,11,3,5), 'hee':(365,2270,11,11,3,6),
    'v':(235,1670,0,7,4,4), 's':(950,6020,11,11,0,9), 'z':(260,5300,0,9,4,7),
    'sh':(800,3100,11,11,2,10), 'b0':(190,835,0,0,3,0),
    'b':(345,1040,5,10,4,5),
    'k':(680,2050,11,11,3,9), 'kb':(595,1450,11,11,3,8),
    'flap':(260,1590,0,0,4,1),
}

score = []

def word(label, notes):
    # Token format: phone, MIDI note, optional loudness offset, optional gate.
    for tok in notes.split():
        key, note, *extra = tok.split('/')
        shift=int(extra[0]) if extra else 0
        gate=extra[1] if len(extra)>1 else 'LEG'
        score.append((label,key,int(note),shift,gate))


def rest(n):
    score.extend([('','-',0,0,'100')]*n)
# Gaps and short releases are spelled into the score, along with every diphthong.
rest(2)
word('Hello', 'h/47 uh/47/-1 l/48 oh/52 oh/52 ow/51 oo/49 oo/47/-2/75')
rest(4)
word('My','m/48 aa/49 ay_mid/50 ih/50')
word('name','n/50 ey/52 ey/53 ii/52 m/50')
word('is','ih/48/-2 z/47/-1')
word('Echo','eh/53 eh/54 eh/53/-1/75')
rest(1)
word('Echo','k/51/0/50 oh/50 oh/49 ow/47 oo/46 oo/45/-2/75')
rest(6)
word("I'm",'aa/47 ay_mid/49 ih/50 m/49')
word('a','uh/47/-2/75')
word('little','l/49 ih/52 ih/51 flap/49 l/47 darkl/47')
word('blue','b0/47 b/48/0/25 l/50 uu/53 uu/53 uu/52/-1')
word('machine','m/49 uh/48/-2 sh/50 sh/51 ii/54 ii/54 ii/52 ii/49 n/47 n/45/-1/75')
rest(6)
word('I','aa/47 ay_mid/49 ih/49')
word('live','l/49 ih/51 ih/50 v/49')
word('in','ih/48/-1 n/47')
word('a','uh/47/-2/75')
word('box','b0/47 b/48/0/25 aa/53 aa/54 aa/52 aa/50/-1/75')
rest(1)
word('box','kb/48/0/25 s/47 s/46/-1/50')
rest(5)
word('I','aa/48 ay_mid/50 ih/49')
word('make','m/48 ey/52 ey/52 ii/51/-1/50')
rest(1)
word('make','k/51/-1/25')
word('music','m/50 y/52 uu/54 uu/53 oo/52 z/51 ih/49 ih/48/-1/50')
rest(1)
word('music','k/47/0/25')
rest(6)
word('Can','k/48/0/50 ae/49 ae/50 n/49')
word('you','y/49 uu/48 uu/47/-1')
word('hear','hee/49 ii/52 ii/53 ih/52 r/51')
word('me','m/50 ii/52 ii/54 ii/55 ii/54 ii/52 ii/50/-1 ii/49/-3/50')
rest(4)


def patch(name,filt,base,sus,q,level,mode):
    return dict(name=name, vco1='saw',vco2='white_noise',vco2_pitch_semitones=0,sync=False,
                vcf_mode=filt,cutoff_hz=base,peak_q=q,
                eg=dict(attack_ms=2,decay_ms=1.5,sustain=sus,release_ms=8),
                vca_mode='EG',vca_level=level,drive=0,portamento_ms=17,
                x='VCF EG INT',y='VCO BALANCE',kaoss=mode,patch_cables=[],mg='unpatched',fx='off')
default_patches = [patch('SYN 1','BPF',415,.265,4.7,.95,'SMOOTH'),
           patch('SYN 2','BPF',2160,.34,6.5,.70,'HOLD')]

def x_position(frequency, patch):
    if patch['x'] == 'VCF CUTOFF':
        position = 11 * math.log(frequency / 30, 16000 / 30)
    else:
        full_depth = 4 * patch['eg']['sustain']
        octave_offset = math.log2(frequency / patch['cutoff_hz'])
        position = 5.5 * (1 + octave_offset / full_depth)
    return max(0, min(11, round(position)))


def make_project(direct=False):
    patches = [dict(p) for p in default_patches]
    if direct:
        for p in patches:
            p['x'] = 'VCF CUTOFF'

    rows = []
    for label, gesture, note, accent, gate in score:
        tracks = []
        for j, patch in enumerate(patches):
            if gesture == '-':
                # Preserve the preceding values while the VCA release decays.
                previous = rows[-1][2+j] if rows else [None, '100', 8, 0, 5, 11]
                settings = [None, '100', *previous[2:]]
            else:
                settings_for_phone = phones[gesture]
                frequency = settings_for_phone[j]
                balance = settings_for_phone[2+j]
                level = settings_for_phone[4+j]
                volume = max(0, min(11, level + accent)) if level else 0
                settings = [note - 2, gate, volume, 0, x_position(frequency, patch), balance]
            tracks.append(settings)
        rows.append([label, gesture, *tracks])
    # Set XY ahead of the first /h/ while both gates are closed.
    first = next(row for row in rows if row[2][0] is not None)
    for row in rows[:2]:
        for j in (2, 3):
            row[j][4:] = first[j][4:]

    # Word boundaries make the patterns easier to enter on a small sequencer.
    boundaries = [0, 14, 25, 41, 52, 68, 81, 94, 108, 124, 136, len(rows)]
    bank = {f'P{i:02}': rows[a:b]
            for i, (a, b) in enumerate(zip(boundaries, boundaries[1:]), start=1)}
    project = dict(title='Echo',text="Hello. My name is Echo. I'm a little blue machine. I live in a box. I make music. Can you hear me?",
                   bpm=222, sample_rate=44100,noise_seed=1031,master_level=.83,
                   model=dict(eg_int_max_octaves=4),
                   patch=patches,row_format=['word (annotation)','gesture (annotation)','SYN 1','SYN 2'],
                   lane_order=['note_midi (null=rest)','gate (% or LEG)','volume (0..11)','pan (-2..2)','x (0..11)','y (0..11)'],
                   song=list(bank),patterns=bank)
    return project

def write_score(project, path):
    # Keep the entire 6-lane setting for both instruments visible on one line.
    header = dict(project)
    bank = header.pop("patterns")
    chunks = []
    for name, pattern in bank.items():
        body = ",\n".join("      " + json.dumps(row) for row in pattern)
        chunks.append('    "' + name + '": [\n' + body + '\n    ]')
    header_text = json.dumps(header, indent=2)[:-2]
    path.write_text(header_text + ',\n  "patterns": {\n' + ",\n".join(chunks) + "\n  }\n}\n", encoding="utf-8")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--direct", action="store_true", help="Comparison using direct X-to-cutoff mapping")
    cli.add_argument("-o", "--output", type=Path)
    args = cli.parse_args()
    project = make_project(args.direct)
    output = args.output or HERE / ("direct_score.json" if args.direct else "echo_score.json")
    write_score(project, output)
    print(output)
