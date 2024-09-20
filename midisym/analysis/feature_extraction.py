import numpy as np

from ..parser.container import Note
from ..parser.container import SymMusicContainer
from ..analysis.utils import get_all_notes


def get_pc_one_hot(all_notes: list[Note]):
    one_hot = np.zeros((len(all_notes), 12))
    pitches = [note.pitch for note in all_notes]
    idx = (np.arange(len(all_notes)), np.remainder(pitches, 12))
    one_hot[idx] = 1

    return one_hot


def get_octave_one_hot(all_notes: list[Note]):
    one_hot = np.zeros((len(all_notes), 10))
    pitches = [note.pitch for note in all_notes]
    idx = (np.arange(len(all_notes)), np.floor_divide(pitches, 12))
    one_hot[idx] = 1

    return one_hot


def get_duration_feature(all_notes: list[Note], ts_beats=4, ticks_per_beat=480):
    ts_beats = 4  # time signature denominator
    ticks_per_beat = 480  # default ticks per beat

    all_notes_durations = [
        (note.end - note.start) / ticks_per_beat / ts_beats for note in all_notes
    ]

    duration_feature = np.expand_dims(
        1 - np.tanh(np.array(all_notes_durations)),
        1,
    )

    return duration_feature


def feature_extraction_gather_all_feat(sym_music_obj: SymMusicContainer):

    all_notes = get_all_notes(sym_music_obj, exclude_drum=True)
    all_notes = sorted(all_notes, key=lambda x: x.start)

    pc_one_hot = get_pc_one_hot(all_notes)
    octave_one_hot = get_octave_one_hot(all_notes)
    ts = (
        sym_music_obj.time_signature_changes[0].denominator
        if len(sym_music_obj.time_signature_changes) != 0
        else 4
    )
    duration_feature = get_duration_feature(
        all_notes,
        ts_beats=ts,
        ticks_per_beat=sym_music_obj.ticks_per_beat,
    )

    feat_0 = np.hstack((duration_feature, pc_one_hot, octave_one_hot))

    return feat_0


def get_pc_one_hot_from_note_array(note_array):
    """Get one-hot encoding of pitch class."""
    one_hot = np.zeros((len(note_array), 12))
    idx = (np.arange(len(note_array)), np.remainder(note_array["pitch"], 12))
    one_hot[idx] = 1
    return one_hot


def get_octave_one_hot_from_note_array(note_array):
    """Get one-hot encoding of octave."""
    one_hot = np.zeros((len(note_array), 10))
    idx = (np.arange(len(note_array)), np.floor_divide(note_array["pitch"], 12))
    one_hot[idx] = 1
    return one_hot


def feature_extraction_score(
    note_array, sym_music_container, score=None, include_meta=False
):
    """Extract features from note_array.
    Parameters
    ----------
    note_array : structured array
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.
    score : partitura score object (optional)
        The partitura score object. If provided, the meta features can be extracted.
    include_meta : bool
        Whether to include meta features. ()

    Returns
    -------
    features : np.array
        level 0 features: duration (1), pitch class one hot (12), octave one hot (10).
        level 1 features: 61 dim
    """
    # Solution for the problem of note tied in make_note_features() but it takes longer to parse each score.
    # if include_meta and isinstance(score, pt.score.Score):
    #     score = pt.score.merge_parts(score.parts)
    #     note_array = score.note_array()
    pc_oh = get_pc_one_hot(note_array)
    octave_oh = get_octave_one_hot(note_array)

    ts_beats = 4  # time signature denominator
    ticks_per_beat = 480  # default ticks per beat
    duration_feature = np.expand_dims(
        1
        - np.tanh(
            (note_array["end"] - note_array["start"]) / ticks_per_beat / ts_beats
        ),
        1,
    )
    # duration is normalized to 1 - tanh(duration / ts_beats) = [-1, 1]

    feat_0, feat_1 = np.hstack((duration_feature, pc_oh, octave_oh)), None
    return feat_0, feat_1
