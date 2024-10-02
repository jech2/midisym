
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
