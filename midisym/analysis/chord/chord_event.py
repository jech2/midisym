from .chord_interval import ChordInterval
from ...parser.utils import parse_chord
import numpy as np

def remove_character(input_string: str, character_to_remove: str) -> str:
    # 입력 문자열에서 특정 문자를 모두 제거하고 반환합니다.
    result_string = input_string.replace(character_to_remove, "")
    return result_string

def pitch_to_chord_polydis_chord_input(pitches, root_number, bass_number):
    chroma = np.zeros(12)
    for pitch in pitches:
        chroma[pitch % 12] = 1
    
    root_one_hot = np.zeros(12)
    root_one_hot[root_number % 12] = 1
    
    bass_one_hot = np.zeros(12)
    bass_one_hot[bass_number % 12] = 1
    
    return np.concatenate([root_one_hot, chroma, bass_one_hot])

def chord_labels_to_one_hot(chd_str, chord_style='pop909'):
    root, quality, bass = parse_chord(chd_str, chord_style=chord_style)
    chord = ChordEvent(root, quality, 0, 16, bass)
    if root not in ['None', None]:
        ci = ChordInterval()
        root_number = ci.pitch_to_midi_pitch(root, 4)
        if bass is not None:
            bass_number = ci.pitch_to_midi_pitch(bass, 4)
        else:
            bass_number = root_number

        pitches = chord.to_pitches(as_name=False)
    
        return pitch_to_chord_polydis_chord_input(pitches, root_number, bass_number)
    else:
        # zeros
        return np.zeros(36)

def chord_name_to_chord_event(chord_name: str, chord_st_idx:int, chord_ed_idx: int):
    if chord_name != 'N':
        root, quality = chord_name.split(':')
        bass = None
        if '/' in quality:
            quality, real_root = quality.split('/')
            chord = ChordEvent(root, quality, chord_st_idx, chord_ed_idx)
            chord_numbers = ['1', '3', '5', '7']
            for t, chord_number in enumerate(chord_numbers):
                if chord_number in real_root:
                    real_root_name = chord.to_pitches(as_name=True)[t]
                    # print(real_root_name)
                    bass = real_root_name
                    break    
    else:
        root = None
        quality = None
        bass = None  

    if root != None:
        return ChordEvent(root, quality, chord_st_idx, chord_ed_idx, bass)
    else:
        return None

def chord_name_to_chroma(chord_name: str, chord_st_idx:int, chord_ed_idx: int):
    chord_event = chord_name_to_chord_event(chord_name, chord_st_idx, chord_ed_idx)
    if chord_event is not None:
        chord_chroma = chord_event.to_chroma()
    else:
        chord_chroma = None
    
    return chord_chroma

class ChordEvent:
    def __init__(self, root, chord_type, start, duration, bass=None):
        self.root = root
        self.chord_type = chord_type
        self.bass = bass
        self.start = start
        self.duration = duration
        
    def __repr__(self):
        return f"ChordEvent({self.chord_type}, start: {self.start}, duration: {self.duration})\n"
    
    def to_tuple(self):
        return (self.root, self.chord_type, self.bass, self.start, self.duration)
    
    def to_dict(self):
        return {
            'root': self.root,
            'chord_type': self.chord_type,
            'bass': self.bass,
            'start': self.start,
            'duration': self.duration
        }
        
    def to_pitches(self, as_name=False):
        ci = ChordInterval()
        root_number = ci.pitch_to_midi_pitch(self.root, 4)
        sym = self.chord_type
        chord_midi_pitches = []
        if self.bass is not None:
            sym = remove_character(sym, "/"+ self.bass)
            bass_number = ci.pitch_to_midi_pitch(self.bass, 3)
            chord_midi_pitches.append(bass_number)

        quality = remove_character(sym, self.root)
        if quality == '':
            quality = 'M'
        chord_midi_pitches += ci.get_chord_pitches(root_number, quality)
        
        if as_name:
            chord_pitch_names = [ci.midi_pitch_to_pitch(pitch) for pitch in chord_midi_pitches]
            return chord_pitch_names
        else:
            return chord_midi_pitches
    
    def to_chroma(self):
        chord_pitches = self.to_pitches()
        chroma_pitches = [chord_pitch % 12 for chord_pitch in chord_pitches]
        
        return chroma_pitches 
    

        
if __name__ == '__main__':
    chord = ChordEvent('C#', 'maj', 0, 4)
    print(chord.to_chroma())