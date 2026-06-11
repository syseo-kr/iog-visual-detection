# ============================================================
# As-run analysis script from the IOG visual-detection study.
# Paths reflect the authors working environment; upstream steps
# (00, 01, 03) require the raw screenshots. To reproduce results
# from the released derived data, use ../reproduce.py instead.
# ============================================================

import os
import base64
import csv
import time
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from PIL import Image
import io

# 환경 변수 로드
load_dotenv()

# 설정 변수
BATCH_SIZE = 10  # 한 번에 처리할 이미지 개수
IMAGE_FOLDER = "img"  # 이미지가 있는 폴더 경로
OUTPUT_CSV = "gambling_results.csv"  # 결과를 저장할 CSV 파일 경로

# LLM 모델 설정
OPENAI_MODEL = "gpt-5-2025-08-07"
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
GEMINI_MODEL = "gemini-2.5-flash"

# 프롬프트 설정
PROMPT = f"""
You will be given a set of image files, each representing a screenshot of a website's landing page. Your task is to determine whether each website appears to be operating as an illegal online gambling site, based solely on the visual content of the image.

Instructions:
• 'O' → The image shows signs of an illegal online gambling website.
• 'X' → The image does not appear to be an illegal online gambling website.
• Use only the visual information in the image for classification.

IMPORTANT: You will be provided with {BATCH_SIZE} images. You MUST analyze ALL {BATCH_SIZE} images provided and include a judgment for EACH one. Do not skip any images.
Make sure you include EVERY filename in your response with its corresponding judgment.

Output Format: 
For each image, provide ONLY the filename and your judgment (O or X) in this exact format:
filename: [exact original filename] judgment: [O or X]

CRITICAL INSTRUCTIONS:
• DO NOT output any additional information other than the output format presented.
• DO NOT change or mix up the name of the image file when providing your judgment.
• DO NOT describe the images or explain your reasoning.
• DO NOT write any descriptions or analysis of what you see in the images.
• ONLY provide the filename and judgment in the exact format specified.

You must analyze exactly these files and include ALL of them in your response:
"""

# API 키 설정
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def encode_image_to_base64(image_path):
    """이미지를 base64로 인코딩"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_to_bytes(image_path):
    """이미지를 바이트로 인코딩"""
    with open(image_path, "rb") as image_file:
        return image_file.read()

def get_all_image_files():
    """이미지 폴더에서 모든 이미지 파일 경로 가져오기"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = []
    
    for file in os.listdir(IMAGE_FOLDER):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(IMAGE_FOLDER, file))
    
    return image_files

def process_images_with_gpt(image_batch, max_retries=3):
    """GPT 모델로 이미지 처리"""
    last_result = ""  # 마지막 응답을 저장할 변수 추가
    
    for attempt in range(max_retries):
        try:
            # 이미지 파일 목록을 먼저 포함시킴
            filenames = [os.path.basename(img) for img in image_batch]
            file_list_text = "\n" + "\n".join(filenames)
            
            messages = [{"role": "system", "content": PROMPT + "\n\n" + file_list_text}]
            
            for img_path in image_batch:
                filename = os.path.basename(img_path)
                base64_image = encode_image_to_base64(img_path)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Image with filename: {filename}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                })
            
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages#,
                #max_tokens=1000
            )
            
            result = response.choices[0].message.content
            last_result = result  # 결과 저장
            
            # 응답 결과 출력
            print("\n----- GPT 응답 결과 (시도 {}/{})) -----".format(attempt+1, max_retries))
            print(result)
            print("----------------------------")
            
            # 응답 유효성 검사: 각 파일에 대한 판단이 있는지 확인
            valid_response = True
            missing_files = []
            for img_path in image_batch:
                filename = os.path.basename(img_path)
                if filename not in result and f"filename: {filename}" not in result:
                    valid_response = False
                    missing_files.append(filename)
            
            if not valid_response:
                print(f"GPT 응답에서 다음 파일에 대한 판단을 찾을 수 없습니다: {', '.join(missing_files)}. 재시도 중... (시도 {attempt+1}/{max_retries})")
            else:
                return result
            
            # 잠시 대기 후 재시도
            time.sleep(2)
            
        except Exception as e:
            print(f"GPT 처리 오류: {e}. 재시도 중... (시도 {attempt+1}/{max_retries})")
            time.sleep(2)
    
    # 최대 재시도 횟수를 초과한 경우
    print(f"GPT 처리 실패: 최대 재시도 횟수({max_retries})를 초과했습니다.")
    return last_result  # 유효하지 않더라도 마지막 응답 반환

def process_images_with_claude(image_batch, max_retries=3):
    """Claude 모델로 이미지 처리"""
    last_result = ""  # 마지막 응답 저장 변수
    
    for attempt in range(max_retries):
        try:
            # 이미지 파일 목록을 먼저 포함시킴
            filenames = [os.path.basename(img) for img in image_batch]
            file_list_text = "\n" + "\n".join(filenames)
            
            messages = [{"role": "user", "content": [
                {"type": "text", "text": PROMPT + "\n\n" + file_list_text},
            ]}]
            
            for img_path in image_batch:
                filename = os.path.basename(img_path)
                with open(img_path, "rb") as img_file:
                    img_data = img_file.read()
                    
                    # 이미지 형식 감지
                    media_type = "image/jpeg"  # 기본값
                    file_ext = os.path.splitext(img_path)[1].lower()
                    if file_ext == ".png":
                        media_type = "image/png"
                    elif file_ext == ".gif":
                        media_type = "image/gif"
                    elif file_ext in [".jpeg", ".jpg"]:
                        media_type = "image/jpeg"
                    elif file_ext == ".webp":
                        media_type = "image/webp"
                    
                    messages[0]["content"].append(
                        {"type": "text", "text": f"Image with filename: {filename}"}
                    )
                    messages[0]["content"].append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.b64encode(img_data).decode('utf-8')
                        }
                    })
            
            response = anthropic_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1000,
                messages=messages
            )
            
            result = response.content[0].text
            last_result = result  # 결과 저장
            
            # 응답 결과 출력 (한 번만 출력)
            print("\n----- Claude 응답 결과 (시도 {}/{})) -----".format(attempt+1, max_retries))
            print(result)
            print("----------------------------")
            
            # 응답 유효성 검사: 각 파일에 대한 판단이 있는지 확인
            valid_response = True
            missing_files = []
            for img_path in image_batch:
                filename = os.path.basename(img_path)
                if filename not in result and f"filename: {filename}" not in result:
                    valid_response = False
                    missing_files.append(filename)
            
            if not valid_response:
                print(f"Claude 응답에서 다음 파일에 대한 판단을 찾을 수 없습니다: {', '.join(missing_files)}. 재시도 중... (시도 {attempt+1}/{max_retries})")
            else:
                return result  # 유효한 응답이면 바로 반환 (추가 출력 없음)
            
            # 잠시 대기 후 재시도
            time.sleep(2)
            
        except Exception as e:
            print(f"Claude 처리 오류: {e}. 재시도 중... (시도 {attempt+1}/{max_retries})")
            time.sleep(2)
    
    # 최대 재시도 횟수를 초과한 경우
    print(f"Claude 처리 실패: 최대 재시도 횟수({max_retries})를 초과했습니다.")
    return last_result  # 유효하지 않더라도 마지막 응답 반환

def process_images_with_gemini(image_batch, max_retries=3):
    """Gemini 모델로 이미지 처리"""
    last_result = ""  # 마지막 응답 저장 변수
    
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            
            # 이미지 파일 목록을 먼저 포함시킴
            filenames = [os.path.basename(img) for img in image_batch]
            file_list_text = "\n" + "\n".join(filenames)
            
            contents = [PROMPT + "\n\n" + file_list_text]
            
            for img_path in image_batch:
                filename = os.path.basename(img_path)
                contents.append(f"Image with filename: {filename}")
                img = Image.open(img_path)
                contents.append(img)
            
            response = model.generate_content(contents)
            
            result = response.text
            last_result = result  # 결과 저장
            
            # 응답 결과 출력
            print("\n----- Gemini 응답 결과 (시도 {}/{})) -----".format(attempt+1, max_retries))
            print(result)
            print("----------------------------")
            
            # 응답 유효성 검사: 각 파일에 대한 판단이 있는지 확인
            valid_response = True
            missing_files = []
            for img_path in image_batch:
                filename = os.path.basename(img_path)
                if filename not in result and f"filename: {filename}" not in result:
                    valid_response = False
                    missing_files.append(filename)
            
            if not valid_response:
                print(f"Gemini 응답에서 다음 파일에 대한 판단을 찾을 수 없습니다: {', '.join(missing_files)}. 재시도 중... (시도 {attempt+1}/{max_retries})")
            else:
                return result
            
            # 잠시 대기 후 재시도
            time.sleep(2)
            
        except Exception as e:
            print(f"Gemini 처리 오류: {e}. 재시도 중... (시도 {attempt+1}/{max_retries})")
            time.sleep(2)
    
    # 최대 재시도 횟수를 초과한 경우
    print(f"Gemini 처리 실패: 최대 재시도 횟수({max_retries})를 초과했습니다.")
    return last_result  # 유효하지 않더라도 마지막 응답 반환

def parse_results(text, filenames):
    """모델 응답에서 파일 이름과 판단 결과 파싱 - 다양한 형식 지원"""
    results = {}
    base_filenames = [os.path.basename(f) for f in filenames]
    
    if not text or text.strip() == "":
        print("응답이 비어있습니다.")
        return results
    
    print("\n=== 파싱 중인 텍스트 ===")
    print(text)
    print("========================\n")
    
    # 1. 일반적인 형식 (filename: xxx judgment: O/X) 처리
    lines = text.strip().split('\n')
    for line in lines:
        if 'judgment:' in line.lower() or 'judgment :' in line.lower():
            parts = line.split('judgment:') if 'judgment:' in line.lower() else line.split('judgment :')
            filename_part = parts[0].strip()
            judgment_part = parts[1].strip() if len(parts) > 1 else ""
            
            # 파일 이름 추출
            filename = ""
            if "filename:" in filename_part.lower():
                filename = filename_part.split("filename:")[1].strip()
            elif "filename :" in filename_part.lower():
                filename = filename_part.split("filename :")[1].strip()
            else:
                # 파일 이름이 명확하게 표시되지 않은 경우 순서대로 맞추기
                for f in base_filenames:
                    if f in filename_part:
                        filename = f
                        break
            
            # 결과에서 파일 이름의 대괄호나 따옴표 제거
            filename = filename.strip("[]'\"\t ")
            
            # 판단 추출 (O 또는 X)
            judgment = ""
            if "O" in judgment_part or "o" in judgment_part:
                judgment = "O"
            elif "X" in judgment_part or "x" in judgment_part:
                judgment = "X"
            
            if filename and judgment:
                results[filename] = judgment
                print(f"파싱 결과 (표준 형식): {filename} -> {judgment}")
    
    # 2. 간단한 형식 (filename.png: O/X) 처리
    if len(results) < len(base_filenames):
        for line in lines:
            if ':' in line and ('O' in line or 'X' in line or 'o' in line or 'x' in line):
                parts = line.split(':')
                if len(parts) >= 2:
                    filename_part = parts[0].strip()
                    judgment_part = parts[1].strip()
                    
                    # 파일명 매칭
                    matched_filename = ""
                    for f in base_filenames:
                        if f in filename_part or filename_part in f:
                            matched_filename = f
                            break
                    
                    if not matched_filename:
                        continue
                    
                    # 판단 추출
                    judgment = ""
                    if "O" in judgment_part or "o" in judgment_part:
                        judgment = "O"
                    elif "X" in judgment_part or "x" in judgment_part:
                        judgment = "X"
                    
                    if matched_filename and judgment and matched_filename not in results:
                        results[matched_filename] = judgment
                        print(f"파싱 결과 (간략 형식): {matched_filename} -> {judgment}")
    
    # 3. 표 형식 처리 (markdown 표나 다른 포맷)
    if len(results) < len(base_filenames):
        for line in lines:
            if '|' in line:
                parts = [part.strip() for part in line.split('|')]
                for part in parts:
                    if not part:
                        continue
                    
                    # 각 셀에서 파일명과 판단 찾기
                    file_match = None
                    for f in base_filenames:
                        if f in part:
                            file_match = f
                            break
                    
                    if not file_match:
                        continue
                    
                    # 판단 추출
                    judgment = ""
                    if "O" in part or "o" in part:
                        judgment = "O"
                    elif "X" in part or "x" in part:
                        judgment = "X"
                    
                    if file_match and judgment and file_match not in results:
                        results[file_match] = judgment
                        print(f"파싱 결과 (표 형식): {file_match} -> {judgment}")
    
    # 4. 목록 형식 처리 (파일명과 판단이 근접해 있는 경우)
    if len(results) < len(base_filenames):
        for i, line in enumerate(lines):
            if not line.strip():
                continue
                
            # 현재 줄에서 파일명 찾기
            file_match = None
            for f in base_filenames:
                if f in line:
                    file_match = f
                    break
            
            if not file_match or file_match in results:
                continue
            
            # 현재 줄이나 다음 줄에서 판단 찾기
            judgment = ""
            search_lines = [line]
            if i + 1 < len(lines):
                search_lines.append(lines[i + 1])
            
            for search_line in search_lines:
                if "O" in search_line or "o" in search_line:
                    judgment = "O"
                    break
                elif "X" in search_line or "x" in search_line:
                    judgment = "X"
                    break
            
            if file_match and judgment:
                results[file_match] = judgment
                print(f"파싱 결과 (목록 형식): {file_match} -> {judgment}")
    
    # 5. 마지막 시도 - 모든 남은 파일에 대해 전체 텍스트에서 판단 추론
    if len(results) < len(base_filenames):
        for f in base_filenames:
            if f not in results:
                # 파일명 주변 컨텍스트 찾기
                context = ""
                for line in lines:
                    if f in line:
                        context = line
                        break
                
                if not context:
                    continue
                
                # 파일명 주변에서 O/X 찾기
                judgment = ""
                if "O" in context or "o" in context:
                    judgment = "O"
                elif "X" in context or "x" in context:
                    judgment = "X"
                
                if judgment:
                    results[f] = judgment
                    print(f"파싱 결과 (컨텍스트 추론): {f} -> {judgment}")
    
    # 6. 완전히 다른 특수 형식 처리 (filename.png: X 형식)
    if len(results) < len(base_filenames):
        for line in lines:
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    filename_part = parts[0].strip()
                    value_part = parts[1].strip()
                    
                    # 파일명 확인
                    if filename_part.endswith('.png') or filename_part.endswith('.jpg') or filename_part.endswith('.jpeg'):
                        for f in base_filenames:
                            if filename_part == f:
                                # 판단 추출
                                judgment = ""
                                if value_part.strip() == 'O' or value_part.strip() == 'o':
                                    judgment = "O"
                                elif value_part.strip() == 'X' or value_part.strip() == 'x':
                                    judgment = "X"
                                
                                if judgment and f not in results:
                                    results[f] = judgment
                                    print(f"파싱 결과 (특수 형식): {f} -> {judgment}")
    
    print(f"최종 파싱 결과 ({len(results)}/{len(base_filenames)}개 파일 처리됨): {results}")
    return results

def create_log_directory():
    """로그 디렉토리 생성"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def create_log_files(timestamp):
    """로그 디렉토리 및 파일 생성"""
    log_dir = create_log_directory()
    
    # 하나의 통합 로그 파일 생성
    combined_log_file = f"{log_dir}/combined_log_{timestamp}.log"
    
    with open(combined_log_file, 'w', encoding='utf-8') as f:
        f.write(f"=== 불법 도박 사이트 탐지 로그 ({timestamp}) ===\n\n")
    
    return log_dir, combined_log_file

def append_to_log(log_file, batch_number, gpt_result, claude_result, gemini_result, gpt_parsed, claude_parsed, gemini_parsed, current_filenames):
    """로그 파일에 결과 추가"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    with open(log_file, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n\n{'='*50}\n")
        log_file.write(f"=== 배치 {batch_number} 로그 ({timestamp}) ===\n")
        log_file.write(f"{'='*50}\n\n")
        log_file.write(f"처리 이미지: {', '.join(current_filenames)}\n\n")
        
        log_file.write("=== GPT 모델 응답 ===\n")
        log_file.write(gpt_result if gpt_result else "응답 없음")
        log_file.write("\n\n")
        
        log_file.write("=== Claude 모델 응답 ===\n")
        log_file.write(claude_result if claude_result else "응답 없음")
        log_file.write("\n\n")
        
        log_file.write("=== Gemini 모델 응답 ===\n")
        log_file.write(gemini_result if gemini_result else "응답 없음")
        log_file.write("\n\n")
        
        log_file.write("=== 파싱 결과 요약 ===\n")
        log_file.write("Filename, GPT, Claude, Gemini\n")
        for filename in current_filenames:
            gpt_judgment = gpt_parsed.get(filename, "")
            claude_judgment = claude_parsed.get(filename, "")
            gemini_judgment = gemini_parsed.get(filename, "")
            log_file.write(f"{filename}, {gpt_judgment}, {claude_judgment}, {gemini_judgment}\n")
        
        # 파싱되지 않은 응답이 있는지 확인
        missing_gpt = [f for f in current_filenames if f not in gpt_parsed]
        missing_claude = [f for f in current_filenames if f not in claude_parsed]
        missing_gemini = [f for f in current_filenames if f not in gemini_parsed]
        
        if missing_gpt or missing_claude or missing_gemini:
            log_file.write("\n=== 누락된 판단 결과 ===\n")
            if missing_gpt:
                log_file.write(f"GPT에서 누락된 파일: {', '.join(missing_gpt)}\n")
            if missing_claude:
                log_file.write(f"Claude에서 누락된 파일: {', '.join(missing_claude)}\n")
            if missing_gemini:
                log_file.write(f"Gemini에서 누락된 파일: {', '.join(missing_gemini)}\n")
    
    print(f"로그가 통합 로그 파일에 추가되었습니다.")

def append_error_to_log(log_file, batch_number, error, current_filenames):
    """오류 정보를 로그 파일에 추가"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n\n{'='*50}\n")
        f.write(f"=== 배치 {batch_number} 오류 로그 ({timestamp}) ===\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"배치 {batch_number} 처리 중 오류 발생: {error}\n")
        f.write(f"처리 이미지: {', '.join(current_filenames)}\n")
    
    print(f"오류 정보가 통합 로그 파일에 추가되었습니다.")

def main():
    # 타임스탬프 생성
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_csv = f"gambling_results_{timestamp}.csv"  # 파일명에 타임스탬프 추가
    
    # 로그 디렉토리 및 파일 생성
    log_dir, combined_log_file = create_log_files(timestamp)
    
    # 모든 이미지 파일 가져오기
    all_images = get_all_image_files()
    total_images = len(all_images)
    
    if total_images == 0:
        print(f"'{IMAGE_FOLDER}' 폴더에 이미지가 없습니다.")
        return
    
    print(f"총 {total_images}개의 이미지를 처리할 수 있습니다.")
    
    # 시작 인덱스 입력 받기
    start_index_input = input(f"몇 번째 이미지부터 시작할까요? (1-{total_images}, 기본값: 1): ").strip()
    start_index = 1
    
    try:
        if start_index_input:
            start_index = int(start_index_input)
            if start_index < 1:
                print("1보다 작은 값은 입력할 수 없습니다. 1부터 시작합니다.")
                start_index = 1
            elif start_index > total_images:
                print(f"{total_images}보다 큰 값은 입력할 수 없습니다. 1부터 시작합니다.")
                start_index = 1
    except ValueError:
        print("유효한 숫자가 아닙니다. 1부터 시작합니다.")
        start_index = 1
    
    # 시작 인덱스를 배치 인덱스로 변환
    batch_start_index = ((start_index - 1) // BATCH_SIZE) * BATCH_SIZE
    
    # 이미 처리된 이미지가 있으면 표시
    if start_index > 1:
        print(f"이미지 {start_index}번부터 처리를 시작합니다. (이미지 1-{start_index-1}번은 건너뜁니다.)")
        
        # 기존 CSV 파일 확인
        continue_csv = input("기존 CSV 파일이 있다면 계속 이어서 작성할까요? (y/n, 기본값: n): ").strip().lower()
        
        if continue_csv == 'y':
            existing_csv = input("기존 CSV 파일 경로를 입력하세요: ").strip()
            if os.path.exists(existing_csv):
                output_csv = existing_csv
                print(f"'{output_csv}' 파일에 결과를 이어서 작성합니다.")
            else:
                print(f"'{existing_csv}' 파일을 찾을 수 없습니다. 새 파일 '{output_csv}'에 결과를 저장합니다.")
                # CSV 파일 초기화
                with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['filename', 'gpt', 'claude', 'gemini'])
        else:
            # CSV 파일 초기화
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['filename', 'gpt', 'claude', 'gemini'])
    else:
        # CSV 파일 초기화
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['filename', 'gpt', 'claude', 'gemini'])
    
    print(f"결과는 '{output_csv}' 파일에 저장됩니다.")
    print(f"로그는 '{combined_log_file}' 파일에 저장됩니다.")
    
    # 배치 단위로 처리
    batch_index = batch_start_index
    while batch_index < total_images:
        # 현재 배치 이미지 선택
        end_index = min(batch_index + BATCH_SIZE, total_images)
        current_batch = all_images[batch_index:end_index]
        current_filenames = [os.path.basename(img) for img in current_batch]
        batch_number = batch_index//BATCH_SIZE + 1
        
        print(f"\n배치 {batch_number} 처리 중... ({batch_index+1}-{end_index}/{total_images})")
        print("현재 처리 중인 이미지:", ", ".join(current_filenames))
        
        try:
            # GPT 모델 처리
            print("\n===== GPT 모델로 처리 중... =====")
            gpt_result = ""
            gpt_retry_count = 0
            max_model_retries = 3
            
            while gpt_retry_count < max_model_retries:
                try:
                    gpt_result = process_images_with_gpt(current_batch)
                    
                    # 결과 파싱 (응답 결과 다시 출력하지 않음)
                    print("\n===== GPT 결과 파싱 중... =====")
                    gpt_parsed = parse_results(gpt_result, current_batch)
                    
                    # 결과 검증
                    if len(gpt_parsed) == len(current_batch):
                        break  # 성공적으로 모든 파일에 대한 결과를 받았으면 종료
                    else:
                        gpt_retry_count += 1
                        if gpt_retry_count < max_model_retries:
                            print(f"GPT 결과가 불완전합니다. GPT만 다시 시도합니다... (시도 {gpt_retry_count}/{max_model_retries})")
                            time.sleep(2)
                        else:
                            print(f"GPT 최대 재시도 횟수({max_model_retries})를 초과했습니다. 현재까지의 결과로 진행합니다.")
                
                except Exception as e:
                    gpt_retry_count += 1
                    print(f"GPT 처리 중 오류 발생: {e}")
                    if gpt_retry_count < max_model_retries:
                        print(f"GPT 다시 시도합니다... (시도 {gpt_retry_count}/{max_model_retries})")
                        time.sleep(2)
                    else:
                        print(f"GPT 최대 재시도 횟수({max_model_retries})를 초과했습니다.")
                        gpt_parsed = {}  # 빈 결과로 설정
            
            # Claude 모델 처리
            print("\n===== Claude 모델로 처리 중... =====")
            claude_result = ""
            claude_retry_count = 0
            
            while claude_retry_count < max_model_retries:
                try:
                    claude_result = process_images_with_claude(current_batch)
                    
                    # 결과 파싱
                    print("\n===== Claude 결과 파싱 중... =====")
                    claude_parsed = parse_results(claude_result, current_batch)
                    
                    # 결과 검증
                    if len(claude_parsed) == len(current_batch):
                        break  # 성공적으로 모든 파일에 대한 결과를 받았으면 종료
                    else:
                        claude_retry_count += 1
                        if claude_retry_count < max_model_retries:
                            print(f"Claude 결과가 불완전합니다. Claude만 다시 시도합니다... (시도 {claude_retry_count}/{max_model_retries})")
                            time.sleep(2)
                        else:
                            print(f"Claude 최대 재시도 횟수({max_model_retries})를 초과했습니다. 현재까지의 결과로 진행합니다.")
                
                except Exception as e:
                    claude_retry_count += 1
                    print(f"Claude 처리 중 오류 발생: {e}")
                    if claude_retry_count < max_model_retries:
                        print(f"Claude 다시 시도합니다... (시도 {claude_retry_count}/{max_model_retries})")
                        time.sleep(2)
                    else:
                        print(f"Claude 최대 재시도 횟수({max_model_retries})를 초과했습니다.")
                        claude_parsed = {}  # 빈 결과로 설정
            
            # Gemini 모델 처리
            print("\n===== Gemini 모델로 처리 중... =====")
            gemini_result = ""
            gemini_retry_count = 0
            
            while gemini_retry_count < max_model_retries:
                try:
                    gemini_result = process_images_with_gemini(current_batch)
                    
                    # 결과 파싱
                    print("\n===== Gemini 결과 파싱 중... =====")
                    gemini_parsed = parse_results(gemini_result, current_batch)
                    
                    # 결과 검증
                    if len(gemini_parsed) == len(current_batch):
                        break  # 성공적으로 모든 파일에 대한 결과를 받았으면 종료
                    else:
                        gemini_retry_count += 1
                        if gemini_retry_count < max_model_retries:
                            print(f"Gemini 결과가 불완전합니다. Gemini만 다시 시도합니다... (시도 {gemini_retry_count}/{max_model_retries})")
                            time.sleep(2)
                        else:
                            print(f"Gemini 최대 재시도 횟수({max_model_retries})를 초과했습니다. 현재까지의 결과로 진행합니다.")
                
                except Exception as e:
                    gemini_retry_count += 1
                    print(f"Gemini 처리 중 오류 발생: {e}")
                    if gemini_retry_count < max_model_retries:
                        print(f"Gemini 다시 시도합니다... (시도 {gemini_retry_count}/{max_model_retries})")
                        time.sleep(2)
                    else:
                        print(f"Gemini 최대 재시도 횟수({max_model_retries})를 초과했습니다.")
                        gemini_parsed = {}  # 빈 결과로 설정
            
            # 로그 파일에 결과 추가
            append_to_log(
                combined_log_file, batch_number, gpt_result, claude_result, gemini_result,
                gpt_parsed, claude_parsed, gemini_parsed, current_filenames
            )
            
            # CSV에 결과 추가
            with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                
                for img_path in current_batch:
                    filename = os.path.basename(img_path)
                    gpt_judgment = gpt_parsed.get(filename, "")
                    claude_judgment = claude_parsed.get(filename, "")
                    gemini_judgment = gemini_parsed.get(filename, "")
                    
                    csv_writer.writerow([filename, gpt_judgment, claude_judgment, gemini_judgment])
                    print(f"CSV에 추가: {filename}, GPT: {gpt_judgment}, Claude: {claude_judgment}, Gemini: {gemini_judgment}")
            
            print(f"배치 {batch_number} 결과가 CSV 파일에 저장되었습니다.")
                
        except Exception as e:
            print(f"전체 처리 중 예상치 못한 오류 발생: {e}")
            # 오류 로그 기록
            append_error_to_log(combined_log_file, batch_number, e, current_filenames)
            
            # 예상치 못한 오류 발생 시에도 계속 진행 (사용자 입력 없이)
            print(f"오류가 발생했지만 다음 배치로 계속 진행합니다.")
        
        # 다음 배치로 이동 (사용자 확인 없이 자동 진행)
        batch_index = end_index
    
    print("\n모든 처리가 완료되었습니다.")
    print(f"결과는 '{output_csv}' 파일에 저장되었습니다.")
    print(f"상세 로그는 '{combined_log_file}' 파일에 저장되었습니다.")

if __name__ == "__main__":
    main()
