from midisym.analysis.utils import get_all_notes
from midisym.analysis.feature_extraction import (
    get_pc_one_hot,
    get_octave_one_hot,
    get_pc_one_hot_from_note_array,
    get_octave_one_hot_from_note_array,
)
import numpy as np


def test_pitch_one_hot(midi_parser):
    all_notes = get_all_notes(midi_parser.sym_music_container, exclude_drum=True)
    all_notes = sorted(all_notes, key=lambda x: x.start)

    note_array = np.array(
        [(n.start, n.end - n.start, n.pitch) for n in all_notes],
        dtype=[("onset_div", "i4"), ("duration_div", "i4"), ("pitch", "i4")],
    )

    one_hot = get_pc_one_hot(all_notes)
    one_hot_np = get_pc_one_hot_from_note_array(note_array)

    assert np.array_equal(one_hot, one_hot_np)


def test_octave_one_hot(midi_parser):
    all_notes = get_all_notes(midi_parser.sym_music_container, exclude_drum=True)
    all_notes = sorted(all_notes, key=lambda x: x.start)

    note_array = np.array(
        [(n.start, n.end - n.start, n.pitch) for n in all_notes],
        dtype=[("onset_div", "i4"), ("duration_div", "i4"), ("pitch", "i4")],
    )

    one_hot = get_octave_one_hot(all_notes)
    one_hot_np = get_octave_one_hot_from_note_array(note_array)

    assert np.array_equal(one_hot, one_hot_np)


# benchmarks
def pitch_one_hot_from_note_array(midi_parser):
    all_notes = get_all_notes(midi_parser.sym_music_container, exclude_drum=True)
    all_notes = sorted(all_notes, key=lambda x: x.start)

    note_array = np.array(
        [(n.start, n.end - n.start, n.pitch) for n in all_notes],
        dtype=[("onset_div", "i4"), ("duration_div", "i4"), ("pitch", "i4")],
    )

    get_pc_one_hot_from_note_array(note_array)


def octave_one_hot_from_note_array(midi_parser):
    all_notes = get_all_notes(midi_parser.sym_music_container, exclude_drum=True)
    all_notes = sorted(all_notes, key=lambda x: x.start)

    note_array = np.array(
        [(n.start, n.end - n.start, n.pitch) for n in all_notes],
        dtype=[("onset_div", "i4"), ("duration_div", "i4"), ("pitch", "i4")],
    )

    get_octave_one_hot_from_note_array(note_array)


def pc_one_hot_from_all_notes(midi_parser):
    all_notes = get_all_notes(midi_parser.sym_music_container, exclude_drum=True)
    all_notes = sorted(all_notes, key=lambda x: x.start)

    get_pc_one_hot(all_notes)


def octave_one_hot_from_all_notes(midi_parser):
    all_notes = get_all_notes(midi_parser.sym_music_container, exclude_drum=True)
    all_notes = sorted(all_notes, key=lambda x: x.start)

    get_octave_one_hot(all_notes)


# benchmarks
# def test_benchmark_pitch_one_hot_from_note_array(benchmark, midi_parser):
#     benchmark(pitch_one_hot_from_note_array, midi_parser)


# def test_benchmark_octave_one_hot_from_note_array(benchmark, midi_parser):
#     benchmark(octave_one_hot_from_note_array, midi_parser)


# def test_benchmark_pc_one_hot_from_all_notes(benchmark, midi_parser):
#     benchmark(pc_one_hot_from_all_notes, midi_parser)


# def test_benchmark_octave_one_hot_from_all_notes(benchmark, midi_parser):
#     benchmark(octave_one_hot_from_all_notes, midi_parser)


from midisym.analysis.feature_extraction import feature_extraction_gather_all_feat


def test_feature_extraction(midi_parser):
    feat_0 = feature_extraction_gather_all_feat(midi_parser.sym_music_container)
    print(feat_0)
    print(feat_0.shape)
