from midisym.analysis.melody_extraction import extract_melody
from midisym.analysis.utils import get_all_notes
from midisym.parser.container import Instrument
from midisym.parser.midi import MidiParser
import copy


def test_melody_extraction(midi_parser):
    sym_music_obj = midi_parser.sym_music_container
    all_notes = get_all_notes(sym_music_obj)
    melody_notes = extract_melody(all_notes)
    print("melody_notes", melody_notes)

    # save the melody notes to a new midi file
    # make a copy of the sym_music_obj
    sym_music_obj_copy = copy.copy(sym_music_obj)
    # make a new instrument
    inst = Instrument(
        program=0,
        name="melody",
    )

    inst.notes = melody_notes

    sym_music_obj_copy.instruments = [inst]

    # dump
    new_midi_parser = MidiParser()
    new_midi_parser.sym_music_obj = sym_music_obj_copy

    new_midi_parser.dump("extracted_melody.mid")
