#!/usr/bin/env python3
"""
여러 영상 파일에서 음성을 추출하고 대본을 생성한 뒤,
전체 내용을 바탕으로 스토리를 추측해 출력하는 CLI 도구.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    source_video: str
    audio_file: str
    language: str
    text: str
    segments: List[Segment]


def run_cmd(command: list[str]) -> None:
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"명령 실행 실패: {' '.join(command)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def ensure_ffmpeg() -> None:
    try:
        run_cmd(["ffmpeg", "-version"])
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg를 찾을 수 없습니다. ffmpeg 설치 후 다시 실행해 주세요."
        ) from exc


def extract_audio(video_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"{video_path.stem}.wav"
    run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
        ]
    )
    return audio_path


def transcribe_audio(audio_path: Path, model_name: str, language: str | None) -> TranscriptResult:
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise RuntimeError(
            "faster-whisper가 설치되지 않았습니다. `pip install -r requirements.txt`를 먼저 실행해 주세요."
        ) from exc

    model = WhisperModel(model_name, device="auto", compute_type="int8")
    segments_iter, info = model.transcribe(str(audio_path), language=language, vad_filter=True)

    segment_list: list[Segment] = []
    full_text_parts: list[str] = []

    for seg in segments_iter:
        cleaned = seg.text.strip()
        if not cleaned:
            continue
        segment_list.append(Segment(start=seg.start, end=seg.end, text=cleaned))
        full_text_parts.append(cleaned)

    full_text = "\n".join(full_text_parts)
    return TranscriptResult(
        source_video="",
        audio_file=str(audio_path),
        language=info.language,
        text=full_text,
        segments=segment_list,
    )


def infer_story_with_openai(text: str, model: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""

    try:
        from openai import OpenAI
    except Exception:
        return ""

    client = OpenAI(api_key=api_key)
    prompt = (
        "다음은 여러 영상에서 추출한 대본입니다.\n"
        "1) 핵심 줄거리(5문장 내외)\n"
        "2) 등장인물/주체\n"
        "3) 갈등/문제\n"
        "4) 결말 또는 메시지\n"
        "를 한국어로 정리해 주세요.\n\n"
        f"대본:\n{text[:120000]}"
    )

    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2,
    )
    return (response.output_text or "").strip()


def infer_story_heuristic(text: str, max_sentences: int = 8) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "대본 내용이 비어 있어 스토리를 추정할 수 없습니다."

    sampled = []
    step = max(1, len(lines) // max_sentences)
    for i in range(0, len(lines), step):
        sampled.append(lines[i])
        if len(sampled) >= max_sentences:
            break

    summary = " ".join(sampled)
    return (
        "[휴리스틱 스토리 추정]\n"
        "대본의 주요 흐름을 기반으로 볼 때, 다음과 같은 스토리로 보입니다:\n"
        f"{summary}\n\n"
        "(정확한 스토리 추정을 원하면 OPENAI_API_KEY를 설정해 LLM 추론을 사용하세요.)"
    )


def save_transcript(result: TranscriptResult, output_path: Path) -> None:
    data = asdict(result)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_video_files(inputs: Iterable[str]) -> list[Path]:
    all_files: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            for ext in ("*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm", "*.m4v"):
                all_files.extend(sorted(p.glob(ext)))
        elif p.is_file():
            all_files.append(p)
        else:
            print(f"[경고] 경로를 찾을 수 없습니다: {raw}", file=sys.stderr)
    # 중복 제거
    uniq = []
    seen = set()
    for f in all_files:
        resolved = str(f.resolve())
        if resolved not in seen:
            seen.add(resolved)
            uniq.append(f)
    return uniq


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="여러 영상 파일의 오디오를 추출하고 대본+스토리 추정을 생성합니다."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="영상 파일 또는 영상 파일들이 들어있는 폴더 경로",
    )
    parser.add_argument("--output-dir", default="outputs", help="결과 저장 폴더")
    parser.add_argument("--model", default="small", help="faster-whisper 모델명 (tiny/base/small/medium/large-v3 등)")
    parser.add_argument("--language", default=None, help="강제 언어 코드 (예: ko, en). 미지정 시 자동 감지")
    parser.add_argument(
        "--story-model",
        default="gpt-4.1-mini",
        help="OPENAI_API_KEY 사용 시 스토리 추론에 사용할 모델명",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    ensure_ffmpeg()

    video_files = collect_video_files(args.inputs)
    if not video_files:
        print("처리할 영상 파일이 없습니다.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    transcript_dir = output_dir / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    merged_text_parts: list[str] = []

    for video in video_files:
        print(f"[진행] 처리 중: {video}")
        audio_path = extract_audio(video, audio_dir)
        result = transcribe_audio(audio_path, args.model, args.language)
        result.source_video = str(video)

        transcript_path = transcript_dir / f"{video.stem}.json"
        save_transcript(result, transcript_path)
        print(f"[완료] 대본 저장: {transcript_path}")

        merged_text_parts.append(f"### 파일: {video.name}\n{result.text}")

    merged_text = "\n\n".join(merged_text_parts)
    merged_path = output_dir / "merged_transcript.txt"
    merged_path.write_text(merged_text, encoding="utf-8")

    story = infer_story_with_openai(merged_text, args.story_model)
    if not story:
        story = infer_story_heuristic(merged_text)

    story_path = output_dir / "story_summary.txt"
    story_path.write_text(story, encoding="utf-8")

    print("\n================ 결과 요약 ================")
    print(f"영상 수: {len(video_files)}")
    print(f"통합 대본: {merged_path}")
    print(f"스토리 요약: {story_path}")
    print("==========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
