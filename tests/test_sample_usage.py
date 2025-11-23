from midisym.parser.renderer import write_audio


def test_chord_event(midi_parser):
    from midisym.analysis.chord.chord_event import chord_labels_to_one_hot
    c = chord_labels_to_one_hot("C:maj", chord_style='pop909')
    print(c, c.shape)

def test_transpose_midi(midi_parser):
    from midisym.analysis.utils import transpose_midi
    transpose_midi(midi_parser.sym_music_container, 'C', 'major')