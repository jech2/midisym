# MIDISym
MIDISym is a Python library that reads, analyses, converts any symbolic music formats (MIDI, XML, ...) into MIR-friendly format. It supports the conversion of representations such as matrix, tokens, and even graphs. It supports MIDI-level analysis such as melody extraction(skyline algorithm), feature extractions such as pitch class, pitch octave, durations. It also contains exporting features such as wav rendering.

## Installation
```bash
pip install midisym

```

## Usage
```
    parser = MidiParser(file_fp)
    
    # parser.sym_music_container.instruments[0].notes[0].pitch
    
    parser.dump_midi()
    
    parser = XMLParser(file_fp)
    
    parser.sym_music_container.instruments[0].notes[0].pitch
    
    parser.dump()
    
    new_sym_container = SymMusicContainer(...)
    
    midi_parser = MidiParser()
    midi_parser.sym_music_container = new_sym_container
    
    midi_parser.dump()
    
    transcribed_midi_parser = TranscribedMidiParser()
```