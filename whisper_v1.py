import whisper
import time
import torch
import os
import gc
import subprocess

# 사용자로부터 입력을 받음
input_folder = input("음성 파일들이 있는 폴더 경로를 입력하세요: ")

# 시작 시간 체크
start_time = time.time()

# GPU 사용 가능한지 확인
if not torch.cuda.is_available():
    raise ValueError("CUDA is not available. This script requires an NVIDIA GPU와 CUDA.")

# GPU 장치 목록 얻기
device_ids = list(range(torch.cuda.device_count()))
print(f"사용 가능한 GPU 장치: {device_ids}")

model_version = 'large-v3'

# 모델 로드 및 GPU에 할당
model = whisper.load_model(model_version)


# transcribe 함수 정의
def transcribe_audio(model, audio_path):
    return model.transcribe(audio_path)

# 폴더 이름에서 필요한 부분을 추출하는 함수
def extract_folder_name(path):
    return os.path.basename(os.path.normpath(path))

# 전체 전사 내용을 하나의 파일에 저장하기 위한 리스트 초기화
all_transcriptions = []

# 입력 폴더에서 파일들을 처리
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(('.mp3', '.wav', '.m4a', '.flac')):
            audio_path = os.path.join(root, file)
            print(f"Processing {audio_path}...")

            # 음성 파일 전사
            result = transcribe_audio(model, audio_path)

            torch.cuda.empty_cache()
            gc.collect()

            # 세그먼트 병합 및 전사 내용 저장
            file_transcription = []
            for segment in result["segments"]:
                text = segment['text']
                file_transcription.append(text)

            # 하나의 파일로 저장하기 위해 추가
            base_filename = os.path.splitext(file)[0]
            all_transcriptions.append(f"[{base_filename}]")
            all_transcriptions.extend(file_transcription)
            all_transcriptions.append("")  # 파일 간 구분을 위한 빈 줄 추가

# 출력 폴더 및 파일 설정
output_folder = input_folder + '_txt'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_filename = f"{extract_folder_name(input_folder)}_v1_result.txt"
output_path = os.path.join(output_folder, output_filename)

# 모든 전사 내용을 하나의 파일로 저장
with open(output_path, "w") as f:
    f.write("\n".join(all_transcriptions))

print(f"전사한 데이터가 {output_path}에 저장되었습니다.")

# 모델 해제 및 메모리 정리
del model
torch.cuda.empty_cache()
gc.collect()

# 종료 시간 체크
end_time = time.time()

# 소요 시간 계산
elapsed_time = end_time - start_time
print('위스퍼 걸린 시간 :', elapsed_time, '초')
