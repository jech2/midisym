import pytest


@pytest.fixture
def midi_parser():
    from midisym.parser.midi import MidiParser

    file_path = "tests/sample/multi-instrumental.mid"
    midi_parser = MidiParser(file_path)
    return midi_parser
