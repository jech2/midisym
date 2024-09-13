import pytest
from midisym.parser.midi import MidiParser


@pytest.fixture
def midi_parser():
    file_path = "tests/sample/multi-instrumental.mid"
    midi_parser = MidiParser(file_path)
    return midi_parser
