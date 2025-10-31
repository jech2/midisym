from midisym.analysis.melody_extraction import save_extracted_melody


def test_melody_extraction(midi_parser):
    save_extracted_melody(
        midi_parser.sym_music_container, "test.mid", do_write_audio=True
    )
