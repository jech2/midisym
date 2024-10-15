import numpy as np
from midisym.parser.container import TempoChange, Marker, Note
from midisym.analysis.grid_quantize import (
    make_grid_quantized_notes,
)
from midisym.constants import PITCH_NAME_TO_ID, PITCH_ID_TO_NAME

def find_nearest_bin(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

# def chord_label_to_ro


class REMILikeCNE:
    def __init__(self):
        
        self.pitch_bins = range(21, 109)    
        self.duration_bins = range(120, 1921, 120)
        self.tempo_bins = np.arange(32, 225, 3)
        self.velocity_bins = np.arange(4, 128, 3)
        self.inst = ['Midi', 'Skyline'] # arrangement and melody
        self.chords = ['+', '/o7', '7', 'M', 'M7', 'm', 'm7', 'o', 'o7', 'sus2', 'sus4'] # 11
        
        self.other_tokens = [
            'Chord_Conti_Conti',
            'Chord_None_None',
            'Tempo_Conti',
            'EOS_None',
        ]
        
        self.chord_mapping = {
            'maj': 'M',
            'min': 'm',
        }

        self.vocab = self.make_vocabulary()
    
    def get_tempo_token(self, tempo):
        if tempo == 'Conti':
            return 'Tempo_Conti'
        else:        
            nearest_tempo = find_nearest_bin(self.tempo_bins, tempo)[0]
            return f'Tempo_{nearest_tempo}'
        
    def get_velocity_token(self, velocity):
        nearest_velocity = find_nearest_bin(self.velocity_bins, velocity)[0]
        return f'Note_Velocity_{nearest_velocity}'
        
    def get_pitch_token(self, pitch):
        assert pitch in self.pitch_bins
        return f'Note_Pitch_{pitch}'
    
    def get_duration_token(self, duration, duration_mode='tick'):
        if duration_mode == 'tick':
            nearest_duration = find_nearest_bin(self.duration_bins, duration)[0]
        elif duration_mode == 'idx':
            nearest_duration = find_nearest_bin(self.duration_bins, duration * 120)[0]
        
        return f'Note_Duration_{nearest_duration}'

    def get_chord_token(self, root, chord):
        if not chord in self.chords:
            if root == 'N' and chord == 'N':
                return 'Chord_None_None'
            if chord in self.chord_mapping:
                chord = self.chord_mapping[chord]
            else:
                NotImplementedError(f'Chord {chord} is not supported')
        return f'Chord_{root}_{chord}'

    def get_bar_token(self):
        return 'Bar_None'
    
    def get_beat_token(self, beat):
        assert beat in range(16)
        return f'Beat_{beat}'
    
    def get_inst_token(self, inst_idx):
        return f'Track_{self.inst[inst_idx]}'

    def make_vocabulary(self):
        vocab = []
        
        ### Bar
        vocab.append(self.get_bar_token())
        
        ### Beat
        for beat in range(16):
            vocab.append(self.get_beat_token(beat))
        
        ### Chord
        for root in range(12):
            for chord in self.chords:
                vocab.append(self.get_chord_token(root, chord))
        
        ### Tempo
        for tempo in self.tempo_bins:
            vocab.append(self.get_tempo_token(tempo))
        
        ### Velocity
        for velocity in self.velocity_bins:
            vocab.append(self.get_velocity_token(velocity))
        
        ### Pitch
        for pitch in self.pitch_bins:
            vocab.append(self.get_pitch_token(pitch))
        
        ### Duration
        for duration in self.duration_bins:
            vocab.append(self.get_duration_token(duration))
        
        ### Track
        for i, _ in enumerate(self.inst):
            vocab.append(self.get_inst_token(i))
                
        ### Other
        for token in self.other_tokens:
            vocab.append(token)
                
        vocab = sorted(vocab)
        vocab = {token: idx for idx, token in enumerate(vocab)}
        
        return vocab
    
    def tokenize_events(self, all_events, grid, chord_style='pop909'):
        """
        tokenize midi events
        if chord_style="pop909", tokenize pop909 style chord labeled midi events

        Args:
            all_events (_type_): all midi events
            grid (_type_): grid calculated from make_grid_quantized_notes

        Raises:
            NotImplementedError: not implemented event

        Returns:
            _type_: token lists
        """
        all_event_seq = []
        prev_bar = -1
        prev_tempo = None
        for i, event in enumerate(all_events):
            if isinstance(event, TempoChange):
                if prev_tempo != event.tempo:
                    all_event_seq.append(self.get_tempo_token(event.tempo))
                else:
                    all_event_seq.append(self.get_tempo_token('Conti'))
                prev_tempo = event.tempo
            elif isinstance(event, Marker):
                if chord_style == 'pop909':
                    root, chord = event.text.split('_')[-1].split(":")
                elif chord_style == 'chorder':
                    root, chord, bass = event.text.split('_')
                    if chord == 'bpm':
                        continue #ignore bpm marker
                all_event_seq.append(self.get_chord_token(PITCH_NAME_TO_ID[root], chord))
            elif isinstance(event, Note):
                if event.end == event.start:
                    print(f'Note {event} has same start and end time. ignored')
                    continue
                start_idx = np.where(grid == event.start)[0][0]
                end_idx = np.where(grid == event.end)[0][0]
                curr_start_bar = start_idx // 16
                curr_pos_start = start_idx % 16
                curr_pos_end = end_idx % 16
                if curr_start_bar != prev_bar:
                    for _ in range(curr_start_bar - prev_bar):
                        all_event_seq.append(self.get_bar_token()) 
                    prev_bar = curr_start_bar
                all_event_seq.append(self.get_beat_token(curr_pos_start))
                all_event_seq.append(self.get_duration_token(end_idx - start_idx, duration_mode="idx")) # max 2 bars
                all_event_seq.append(self.get_pitch_token(event.pitch))
            else:
                raise NotImplementedError(f'Event {event} is not supported')
        
        return all_event_seq
        
    def make_all_event_list(self, sym_obj, sym_data_type):
        # quantize the notes
        q_sym_obj, grid = make_grid_quantized_notes(
            sym_obj=sym_obj,
            sym_data_type=sym_data_type,
        )
        
        # chord as marker
        all_events = q_sym_obj.markers + q_sym_obj.tempo_changes

        for inst in q_sym_obj.instruments:
            all_events.append(inst.notes)
        # 시작 시간이 같으면 tempo, marker, note 순으로 정렬
        all_events = sorted(
            all_events, 
            key=lambda x: (
                getattr(x, 'start', getattr(x, 'time', 0)),  # start가 있으면 사용, 없으면 time, 둘 다 없으면 0
                isinstance(x, Note),    # note를 마지막으로 정렬
                isinstance(x, Marker), # marker를 다음으로 정렬
                isinstance(x, TempoChange),  # tempo를 우선으로 정렬
            )
        )
        
        return all_events, grid
        
    def tokenize(self, sym_obj, sym_data_type, chord_style):
        all_events, grid = self.make_all_event_list(sym_obj, sym_data_type)
        token_seq = self.tokenize_events(all_events, grid, chord_style)
        return token_seq
        
    def token_to_idx(self, token):
        return self.vocab[token]
    
    def idx_to_token(self, idx):
        return list(self.vocab.keys())[idx]
    
    def __len__(self):
        return len(self.vocab)
    
    def word_to_sym_obj(self, word_seq, bar_resolution=16, ticks_per_beat=480):
        tick_per_pos = ticks_per_beat // (bar_resolution // 4)
        current_tick = 0
        
        from midisym.parser.container import SymMusicContainer, TempoChange, Marker, Note, Instrument
        sym_obj = SymMusicContainer(ticks_per_beat=ticks_per_beat)
        sym_obj.instruments.append(Instrument())
        
        current_pitch = None
        current_duration = None
        current_velocity = None

        current_bar = -1
        current_pos = 0
        
        for i, token in enumerate(word_seq):
            split_word = token.split('_')
            token_type = split_word[0]
            if token_type == "Bar":
                current_bar += 1
                current_pos = 0
                current_tick = current_bar * ticks_per_beat * 4 + current_pos * tick_per_pos
            elif token_type == "Beat":
                n_pos = int(split_word[1])
                current_pos = n_pos
                current_tick = current_bar * ticks_per_beat * 4 + current_pos * tick_per_pos
            elif token_type == "Tempo":
                tempo = int(split_word[1])
                time = current_tick
                sym_obj.tempo_changes.append(TempoChange(time=time, tempo=tempo))
            elif token_type == "Chord":
                if split_word[1] == "None":
                    root = "N"
                else:
                    root = PITCH_ID_TO_NAME[int(split_word[1])]
                quality = split_word[2]
                sym_obj.markers.append(Marker(time=current_tick, text=f"{root}_{quality}"))
            elif token_type == "Note":
                note_event_type = split_word[1]
                if note_event_type == "Pitch":
                    current_pitch = int(split_word[2])
                elif note_event_type == "Duration":
                    current_duration = int(split_word[2])
                elif note_event_type == "Velocity":
                    current_velocity = int(split_word[2])
                    if current_pitch is not None and current_duration is not None and current_velocity is not None:
                        sym_obj.instruments[0].notes.append(
                            Note(
                                start=current_tick,
                                end=current_tick + current_duration,
                                pitch=current_pitch,
                                velocity=current_velocity,
                            )
                        )
                        current_pitch = None
                        current_duration = None
                        current_velocity = None
                    else:
                        raise ValueError("Note event is not complete")
            
        return sym_obj
    
    def word_to_midi(self, word_seq, out_fp, bar_resolution=16, tick_per_beat=480):
        sym_obj = self.word_to_sym_obj(word_seq, bar_resolution, tick_per_beat)
        from midisym.parser.midi import MidiParser
        parser = MidiParser(sym_music_container=sym_obj)
        parser.dump(out_fp)
        
    def make_inst_events(self, q_sym_obj, grid, inst_idx, use_tempo_changes=True, bar_resolution=16):    
        grid_list = [[] for _ in range(len(grid))]
        
        if use_tempo_changes:
            for tempo in q_sym_obj.tempo_changes:
                start_idx = np.where(grid == tempo.time)[0][0]
                grid_list[start_idx].append(tempo)
        
        # map the tempo and chord to the grid
        for marker in q_sym_obj.markers:
            if 'global_bpm' in marker.text:
                continue
            start_idx = np.where(grid == marker.time)[0][0]
            grid_list[start_idx].append(marker)
        
        for note in q_sym_obj.instruments[inst_idx].notes:
            start_idx = np.where(grid == note.start)[0][0]
            end_idx = np.where(grid == note.end)[0][0]
            grid_list[start_idx].append(note)
        
        return grid_list
    
    
    def tokenize_inst_events(self, events, inst_idx, bar_resolution=16, chord_style="pop909"):
        tokenized_seq = []
        prev_tempo = None
        
        for i in range(0, len(events), bar_resolution):
            bar_events = events[i:i+bar_resolution]
            current_bar = i // bar_resolution
            tokenized_seq.append(self.get_inst_token(inst_idx))
            tokenized_seq.append(self.get_bar_token())
            for j, db_event in enumerate(bar_events):
                if len(db_event) > 0:
                    tokenized_seq.append(self.get_beat_token(j))
                for event in db_event: 
                    if isinstance(event, TempoChange):
                        if prev_tempo != event.tempo:
                            tokenized_seq.append(self.get_tempo_token(event.tempo))
                        else:
                            tokenized_seq.append(self.get_tempo_token('Conti'))
                        prev_tempo = event.tempo
                    elif isinstance(event, Marker):
                        if chord_style == 'pop909':
                            root, chord = event.text.split('_')[-1].split(":")
                        elif chord_style == 'chorder':
                            root, chord, bass = event.text.split('_')
                        tokenized_seq.append(self.get_chord_token(PITCH_NAME_TO_ID[root], chord))
                    elif isinstance(event, Note):
                        tokenized_seq.append(self.get_pitch_token(event.pitch))
                        tokenized_seq.append(self.get_duration_token(event.end - event.start, duration_mode='tick'))
                        tokenized_seq.append(self.get_velocity_token(event.velocity))
                    else:
                        raise NotImplementedError(f'Event {event} is not supported')
        bar_starts = np.where(np.array(tokenized_seq) == self.get_bar_token())[0]
        bar_idxs = [(int(bar_starts[i]), int(bar_starts[i+1])) for i in range(len(bar_starts)-1)]
        # last bar
        bar_idxs.append((int(bar_starts[-1]), len(tokenized_seq)))
        
        print(tokenized_seq)
        
        tokenized_seq = [self.token_to_idx(token) for token in tokenized_seq]
        
        return tokenized_seq, bar_idxs