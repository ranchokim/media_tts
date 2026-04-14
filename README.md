# media_tts

여러 개의 긴 영상(예: 1시간 내외)에서 음성을 추출해 대본을 만들고,
전체 대본을 기반으로 영상들의 스토리를 추정하는 Python CLI 프로그램입니다.

## 기능

- 여러 영상 파일/폴더를 한 번에 입력
- `ffmpeg`로 오디오 추출 (16kHz mono wav)
- `faster-whisper`로 대본 생성
- 전체 대본 통합
- 스토리 추정
  - `OPENAI_API_KEY`가 있으면 LLM(`--story-model`) 사용
  - 없으면 휴리스틱 요약 사용

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

그리고 시스템에 `ffmpeg`가 설치되어 있어야 합니다.

## 사용법

```bash
python media_story_analyzer.py /path/to/video1.mp4 /path/to/video2.mkv --output-dir outputs --model small --language ko
```

또는 폴더를 통째로 입력할 수 있습니다.

```bash
python media_story_analyzer.py /path/to/videos_dir --output-dir outputs
```

## 결과물

- `outputs/audio/*.wav`: 추출된 오디오
- `outputs/transcripts/*.json`: 파일별 대본/세그먼트
- `outputs/merged_transcript.txt`: 전체 통합 대본
- `outputs/story_summary.txt`: 스토리 추정 결과

## LLM 기반 스토리 추정 사용 (선택)

```bash
export OPENAI_API_KEY="your_api_key"
python media_story_analyzer.py /path/to/videos_dir --story-model gpt-4.1-mini
```

## 참고

- 1시간 분량 영상이 여러 개인 경우 처리 시간이 오래 걸릴 수 있습니다.
- GPU가 있으면 Whisper 추론 속도가 빨라질 수 있습니다.
- 모델 크기는 정확도/속도 트레이드오프입니다.
