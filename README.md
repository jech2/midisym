# MIDISym
MIDISym is a Python library that reads, analyses, converts any symbolic music formats (MIDI, XML, ...) into MIR-friendly format. It supports the conversion of representations such as matrix, tokens, and even graphs. It supports MIDI-level analysis such as melody extraction(skyline algorithm), feature extractions such as pitch class, pitch octave, durations. It also contains exporting features such as wav rendering.

## Features
- MIDI parsing: supporting mido (basic) and symusic (fast)
- MIDI analysis: melody extraction, feature extractions 
- MIDI conversion: symbolic music representations (matrix, tokens, graphs)
- MIDI exporting: (.mid, .wav)

## Installation
```bash
pip install .

```

## Usage
```
    import midisym

    parser = midisym.parser.midi.MidiParser(file_fp)
    # or, simply
    parser = midisym.read_midi(file_fp)

    # parser.sym_music_container.instruments[0].notes[0].pitch
    
    parser.dump()
```
