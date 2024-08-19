import whisper
import time
import torch
import os
import gc
import subprocess
import sys

input_folder = input("음성 파일들이 있는 폴더 경로를 입력하세요: ")
start_time = time.time()

# GPU 사용 가능한지 확인
if not torch.cuda.is_available():
    raise ValueError("CUDA is not available. This script requires an NVIDIA GPU와 CUDA.")

device_ids = list(range(torch.cuda.device_count()))
print(f"사용 가능한 GPU 장치: {device_ids}")

model_version = 'large-v3'

# 모델 로드 및 GPU에 할당
model = whisper.load_model(model_version).to(f"cuda:{device_ids[0]}")

# 모델을 DataParallel로 감싸기
model = torch.nn.DataParallel(model, device_ids=device_ids)

# ffmpeg를 사용하여 오디오 파일을 16kHz 모노로 변환하고 소리 크기 조정 및 무음 제거하는 함수
def process_and_convert_audio(audio_path):
    converted_path = "./converted.wav"
    processed_path = "./processed.wav"

    # 1단계: 오디오 파일을 16kHz 모노로 변환
    command_convert = [
        "ffmpeg", "-i", audio_path,
        "-ar", "16000", "-ac", "1",
        converted_path
    ]
    subprocess.run(command_convert, check=True)

    # 2단계: 소리 크기 조정 및 무음 부분 제거
    command_process = [
        "ffmpeg", "-i", converted_path,
        "-af",
        "loudnorm=I=-24:TP=-1.5:LRA=11:measured_I=-24:measured_LRA=7:measured_TP=-2:measured_thresh=-34:offset=10,"
        "silenceremove=start_periods=1:start_threshold=-80dB:start_duration=1.5:stop_periods=-1:stop_threshold=-60dB",
        "-y",  # 기존 파일 덮어쓰기 허용
        processed_path
    ]
    subprocess.run(command_process, check=True)

    # 중간 변환 파일 삭제
    os.remove(converted_path)

    return processed_path

# 폴더 이름에서 필요한 부분을 추출하는 함수
def extract_folder_name(path):
    return os.path.basename(os.path.normpath(path))

# transcribe 함수 정의
def transcribe_audio(model, audio_path):
    return model.module.transcribe(audio_path, patience=5.0, beam_size=5)

# 반복된 문장을 제거하는 함수
def remove_repeated_sentences(transcription):
    filtered_transcription = []
    previous_sentence = None
    for sentence in transcription:
        if sentence != previous_sentence:
            filtered_transcription.append(sentence)
            previous_sentence = sentence
    return filtered_transcription

# 전체 전사 내용을 하나의 파일에 저장하기 위한 리스트 초기화
all_transcriptions = []

# 입력 폴더에서 파일들을 처리
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(('.mp3', '.wav', '.m4a', '.flac')):
            audio_path = os.path.join(root, file)
            print(f"Processing {audio_path}...")

            # 오디오 파일을 처리 및 변환
            processed_audio_path = process_and_convert_audio(audio_path)

            # 음성 파일 전사
            result = transcribe_audio(model, processed_audio_path)

            # 메모리 정리 및 해제
            os.remove(processed_audio_path)
            del processed_audio_path

            torch.cuda.empty_cache()
            gc.collect()

            # 세그먼트 병합 및 전사 내용 저장
            file_transcription = []
            for segment in result["segments"]:
                text = segment['text']
                file_transcription.append(text)

            # 중복 문장 제거
            filtered_transcription = remove_repeated_sentences(file_transcription)

            # 하나의 파일로 저장하기 위해 추가
            base_filename = os.path.splitext(file)[0]
            all_transcriptions.append(f"[{base_filename}]")
            all_transcriptions.extend(filtered_transcription)
            all_transcriptions.append("")  # 파일 간 구분을 위한 빈 줄 추가

# 출력 폴더 및 파일 설정
output_folder = input_folder + '_txt'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 실행 파일 이름에서 앞의 두 글자를 추출
def get_name():
    executable_name = os.path.basename(sys.argv[0])
    return executable_name[:2]

output_filename = f"{extract_folder_name(input_folder)}_v2_result.txt"
output_path = os.path.join(output_folder, output_filename)

# 모든 전사 내용을 하나의 파일로 저장
with open(output_path, "w") as f:
    f.write("\n".join(all_transcriptions))

print(f"전사한 데이터가 {output_path}에 저장되었습니다.")

# 모델 해제 및 메모리 정리
del model
torch.cuda.empty_cache()
gc.collect()

end_time = time.time()
elapsed_time = end_time - start_time
print('위스퍼 걸린 시간 :', elapsed_time, '초')
