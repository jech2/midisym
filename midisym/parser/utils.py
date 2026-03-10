from .container import SymMusicContainer
import numpy as np
import copy


def _dedupe_time_signatures(time_signature_changes):
    if not time_signature_changes:
        return []

    sorted_changes = sorted(time_signature_changes, key=lambda ts: (ts.time, ts.numerator, ts.denominator))
    deduped = []
    for ts in sorted_changes:
        if deduped and (
            deduped[-1].time == ts.time
            and deduped[-1].numerator == ts.numerator
            and deduped[-1].denominator == ts.denominator
        ):
            continue
        deduped.append(ts)
    return deduped


def _bar_start_ticks(sym_obj: SymMusicContainer, max_bar_index: int) -> list[int]:
    tpb = sym_obj.ticks_per_beat
    ts_changes = _dedupe_time_signatures(sym_obj.time_signature_changes)
    if not ts_changes or ts_changes[0].time != 0:
        default_ts = copy.deepcopy(ts_changes[0]) if ts_changes else None
        if default_ts is None:
            from .container import TimeSignature

            default_ts = TimeSignature(4, 4, 0)
        else:
            default_ts.time = 0
        ts_changes = [default_ts] + ts_changes

    target_tick = max(sym_obj.max_tick, tpb * 4 * max_bar_index)
    bar_starts = [0]

    for idx, ts in enumerate(ts_changes):
        next_change_tick = ts_changes[idx + 1].time if idx + 1 < len(ts_changes) else None
        ticks_per_bar = int(round(tpb * 4 * ts.numerator / ts.denominator))
        current_tick = max(ts.time, bar_starts[-1])

        if current_tick > bar_starts[-1]:
            bar_starts.append(current_tick)

        while len(bar_starts) <= max_bar_index:
            next_bar_tick = current_tick + ticks_per_bar
            if next_change_tick is not None and next_bar_tick > next_change_tick:
                break
            bar_starts.append(next_bar_tick)
            current_tick = next_bar_tick

        if len(bar_starts) > max_bar_index:
            break

    if len(bar_starts) <= max_bar_index:
        last_ts = ts_changes[-1]
        ticks_per_bar = int(round(tpb * 4 * last_ts.numerator / last_ts.denominator))
        current_tick = bar_starts[-1]
        while len(bar_starts) <= max_bar_index:
            current_tick += ticks_per_bar
            bar_starts.append(current_tick)

    return bar_starts

def seconds_to_ticks(seconds, ticks_per_beat, tempo, tempo_ref='microseconds'):
    """Convert seconds to MIDI ticks."""

    # mido tempo is in microseconds per beat
    if tempo_ref == "microseconds":   
        microseconds_per_tick = tempo / ticks_per_beat
        ticks = (seconds * 1000000) / microseconds_per_tick
    elif tempo_ref == 'BPM':
        ticks = (seconds * tempo) / 60 / ticks_per_beat
    else:
        raise ValueError(f"tempo_ref must be 'microseconds' or 'BPM'. Got {tempo_ref}")
    return int(ticks)

def get_ticks_to_seconds_grid(midi_obj: SymMusicContainer) -> np.array:
    # max_tick = max([note.end for inst in midi_obj.instruments for note in inst.notes])
    # max_tick = max(max_tick, max([note.start for inst in midi_obj.instruments for note in inst.notes]))
    # max_tick = max(max_tick, midi_obj.max_tick)
    
    ticks_to_seconds = np.zeros(midi_obj.max_tick + 1)
    ticks_per_beat = midi_obj.ticks_per_beat
    tempo_changes = midi_obj.tempo_changes  # Assumed to be a list of (tick, bpm) tuples

    # Initialize time tracking variables
    accumulated_time_seconds = 0.0
    current_tempo_index = 0
    current_tempo_change = tempo_changes[current_tempo_index]
    current_bpm = current_tempo_change.tempo
    seconds_per_beat = 60.0 / current_bpm
    ticks_per_second = ticks_per_beat / seconds_per_beat
    
    # Iterate through each tick up to the maximum tick value
    final_tick = midi_obj.max_tick

    for tick in range(final_tick + 1):
        # Update tempo if we reach a new tempo change tick
        if current_tempo_index < len(tempo_changes) - 1 and tick == tempo_changes[current_tempo_index + 1].time:
            current_tempo_index += 1
            current_tempo_change = tempo_changes[current_tempo_index]
            current_bpm = current_tempo_change.tempo
            seconds_per_beat = 60.0 / current_bpm
            ticks_per_second = ticks_per_beat / seconds_per_beat

        # Store accumulated time for the current tick
        ticks_to_seconds[tick] = accumulated_time_seconds

        # Advance accumulated time by 1 tick in seconds
        accumulated_time_seconds += 1 / ticks_per_second

    return ticks_to_seconds


def parse_chord(chord_text, chord_style):
    bass = None
    try:
        if chord_style == 'pop909':
            if chord_text == 'N':
                return None, None, None
            root, chord = chord_text.split('_')[-1].split(":")
            if '/' in chord:
                chord, _ = chord.split('/')
                # TODO: handle bass
            if '(' in chord:
                chord, _ = chord.split('(')
        elif chord_style == 'chorder':
            root, chord, bass = chord_text.split('_')
        elif chord_style == 'maj_min':
            # majmin style
            if 'maj' in chord_text:
                root = chord_text.split('maj')[0]
                chord = 'M'
            elif 'min' in chord_text:
                root = chord_text.split('min')[0]
                chord = 'm'
    except:
        raise ValueError(f"Cannot parse chord {chord_text} with style {chord_style}")
    return root, chord, bass

def resample_ticks_per_beat(sym_obj: SymMusicContainer, ticks_per_beat: int):
    """ 
    resample the ticks per beat of a sym_obj and all its notes, tempo changes, and markers
    """
    ori_ticks_per_beat = sym_obj.ticks_per_beat
    sym_obj.ticks_per_beat = ticks_per_beat
    # note
    for inst in sym_obj.instruments:
        for note in inst.notes:
            note.start = note.start * ticks_per_beat // ori_ticks_per_beat
            note.end = note.end * ticks_per_beat // ori_ticks_per_beat
    # tempo
    for tempo in sym_obj.tempo_changes:
        tempo.time = tempo.time * ticks_per_beat // ori_ticks_per_beat
    
    # marker
    for marker in sym_obj.markers:
        marker.time = marker.time * ticks_per_beat // ori_ticks_per_beat
        
    # max tick
    sym_obj.max_tick = sym_obj.max_tick * ticks_per_beat // ori_ticks_per_beat
    
    return sym_obj 

def crop_midi_obj(midi_parser, start, end, select_inst=None, unit='tick'):
    if unit == 'tick':
        start_tick = start
        end_tick = end
    elif unit == 'bar':
        bar_starts = _bar_start_ticks(midi_parser.sym_music_container, end)
        start_tick = bar_starts[start]
        end_tick = bar_starts[end]
    else:
        raise ValueError(f"unit must be 'tick' or 'bar'. Got {unit}")

    new_parser = copy.deepcopy(midi_parser)
    if select_inst is not None:
        new_parser.sym_music_container.instruments = [new_parser.sym_music_container.instruments[i] for i in select_inst]

    new_midi_obj = new_parser.sym_music_container
    
    # Keep any note overlapping the crop window and clip to the boundaries.
    for i, instrument in enumerate(new_midi_obj.instruments):
        cropped_notes = []
        for note in instrument.notes:
            if note.end <= start_tick or note.start >= end_tick:
                continue
            new_note = copy.deepcopy(note)
            new_note.start = max(note.start, start_tick) - start_tick
            new_note.end = min(note.end, end_tick) - start_tick
            if new_note.end > new_note.start:
                cropped_notes.append(new_note)
        new_midi_obj.instruments[i].notes = cropped_notes
            
    # marker related events only
    selected_markers = [marker for marker in new_midi_obj.markers if start_tick <= marker.time < end_tick]
    for marker in selected_markers:
        marker.time -= start_tick

    # 맨 처음 마커가 없으면 추가
    if len(selected_markers) > 0 and selected_markers[0].time > new_midi_obj.ticks_per_beat / 2:
        for marker in new_midi_obj.markers:
            if marker.time < selected_markers[0].time:
                first_marker = marker
                first_marker.time = 0
                break

        selected_markers.insert(0, first_marker)
    new_midi_obj.markers = selected_markers
    new_midi_obj.max_tick = (end_tick - start_tick)
        
    return new_parser
