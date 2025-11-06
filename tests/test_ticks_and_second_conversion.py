from midisym.parser.utils import get_ticks_to_seconds_grid
from midisym.parser.container import Note, Instrument

def test_ticks_to_seconds_grid(midi_parser):
    grid = get_ticks_to_seconds_grid(midi_parser.sym_music_container)
    print(grid[-50:])
    
def test_ticks_to_seconds_grid_2(analyzed_performance_midi_parser):
    grid = get_ticks_to_seconds_grid(analyzed_performance_midi_parser.sym_music_container)
    print(grid[-50:])

def test_extract_beats_downbeats(midi_parser_mpag_score_midi):
    from midisym.analysis.utils import extract_beats_downbeats
    beats, downbeats = extract_beats_downbeats(midi_parser_mpag_score_midi.sym_music_container)

    import librosa
    import soundfile as sf
    import numpy as np

    def create_beat_audio(beats, downbeats, duration, output_file='beats.wav', sr=22050):
        """
        Beat와 downbeat를 클릭 사운드로 변환하여 WAV 파일 생성
        
        Args:
            beats: beat 시간 리스트 (초)
            downbeats: downbeat 시간 리스트 (초)
            duration: 오디오 총 길이 (초)
            output_file: 출력 파일명
            sr: 샘플레이트
        """
        # Beat 클릭 (1000Hz, 낮은 볼륨)
        beat_clicks = librosa.clicks(
            times=beats,
            sr=sr,
            length=int(duration * sr),
            click_freq=1000.0,  # 1000Hz
            click_duration=0.1   # 0.1초
        )
        
        # Downbeat 클릭 (2000Hz, 높은 볼륨) - 강조
        downbeat_clicks = librosa.clicks(
            times=downbeats,
            sr=sr,
            length=int(duration * sr),
            click_freq=2000.0,   # 2000Hz (더 높은 음)
            click_duration=0.15  # 0.15초 (더 길게)
        )
        
        # 믹스 (downbeat를 더 크게)
        audio = beat_clicks * 0.5 + downbeat_clicks * 1.0
        
        # 정규화
        audio = audio / np.max(np.abs(audio))
        
        # WAV 파일로 저장
        sf.write(output_file, audio, sr)
        print(f"클릭 트랙 저장됨: {output_file}")
        
        return audio

    # 마지막 beat 시간 + 여유
    duration = beats[-1] + 2.0

    # WAV 파일 생성
    create_beat_audio(
        beats=beats,
        downbeats=downbeats,
        duration=duration,
        output_file='beat_clicks.wav'
    )