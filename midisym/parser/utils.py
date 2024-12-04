from .container import SymMusicContainer
from typing import Dict
import numpy as np

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

    