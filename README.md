# MIDISym
This is a simple Python library that reads MIDI files using the `mido` library.

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