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

# Load environment variables
load_dotenv()

# Configuration
BATCH_SIZE = 10  # Number of images per batch
IMAGE_FOLDER = "img"  # Path to the folder containing the images
# NOTE: this folder holds the STANDARDIZED screenshots (224x224, bicubic, JPEG q90,
# identical to 01_extract_features.py) -- the LMMs receive provenance-standardized
# input, the same image given to CLIP and the handcrafted-feature pipeline.
# See README, 'Notes on inputs and protocol'.
OUTPUT_CSV = "gambling_results.csv"  # Path to the CSV file where results are saved

# LLM model settings
OPENAI_MODEL = "gpt-5-2025-08-07"
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
GEMINI_MODEL = "gemini-2.5-flash"

# Prompt definition
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

# API key setup
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def encode_image_to_base64(image_path):
    """Encode an image as base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_to_bytes(image_path):
    """Read an image as raw bytes."""
    with open(image_path, "rb") as image_file:
        return image_file.read()

def get_all_image_files():
    """Return the paths of all image files in the image folder."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_files = []
    
    for file in os.listdir(IMAGE_FOLDER):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(IMAGE_FOLDER, file))
    
    return image_files

def process_images_with_gpt(image_batch, max_retries=3):
    """Run the GPT model on a batch of images."""
    last_result = ""  # Variable holding the most recent response
    
    for attempt in range(max_retries):
        try:
            # Include the list of image filenames first
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
            last_result = result  # Save the result
            
            # Print the response
            print("\n----- GPT response (attempt {}/{}) -----".format(attempt+1, max_retries))
            print(result)
            print("----------------------------")
            
            # Validate the response: confirm a judgment is present for each file
            valid_response = True
            missing_files = []
            for img_path in image_batch:
                filename = os.path.basename(img_path)
                if filename not in result and f"filename: {filename}" not in result:
                    valid_response = False
                    missing_files.append(filename)
            
            if not valid_response:
                print(f"No judgment found in the GPT response for the following files: {', '.join(missing_files)}. Retrying... (attempt {attempt+1}/{max_retries})")
            else:
                return result
            
            # Wait briefly, then retry
            time.sleep(2)
            
        except Exception as e:
            print(f"GPT error: {e}. Retrying... (attempt {attempt+1}/{max_retries})")
            time.sleep(2)
    
    # Maximum number of retries exceeded
    print(f"GPT failed: exceeded the maximum number of retries ({max_retries}).")
    return last_result  # return the most recent response even if it is invalid

def process_images_with_claude(image_batch, max_retries=3):
    """Run the Claude model on a batch of images."""
    last_result = ""  # Variable holding the most recent response
    
    for attempt in range(max_retries):
        try:
            # Include the list of image filenames first
            filenames = [os.path.basename(img) for img in image_batch]
            file_list_text = "\n" + "\n".join(filenames)
            
            messages = [{"role": "user", "content": [
                {"type": "text", "text": PROMPT + "\n\n" + file_list_text},
            ]}]
            
            for img_path in image_batch:
                filename = os.path.basename(img_path)
                with open(img_path, "rb") as img_file:
                    img_data = img_file.read()
                    
                    # Detect the image format
                    media_type = "image/jpeg"  # default
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
            last_result = result  # Save the result
            
            # Print the response (only once)
            print("\n----- Claude response (attempt {}/{}) -----".format(attempt+1, max_retries))
            print(result)
            print("----------------------------")
            
            # Validate the response: confirm a judgment is present for each file
            valid_response = True
            missing_files = []
            for img_path in image_batch:
                filename = os.path.basename(img_path)
                if filename not in result and f"filename: {filename}" not in result:
                    valid_response = False
                    missing_files.append(filename)
            
            if not valid_response:
                print(f"No judgment found in the Claude response for the following files: {', '.join(missing_files)}. Retrying... (attempt {attempt+1}/{max_retries})")
            else:
                return result  # valid response: return immediately (no extra output)
            
            # Wait briefly, then retry
            time.sleep(2)
            
        except Exception as e:
            print(f"Claude error: {e}. Retrying... (attempt {attempt+1}/{max_retries})")
            time.sleep(2)
    
    # Maximum number of retries exceeded
    print(f"Claude failed: exceeded the maximum number of retries ({max_retries}).")
    return last_result  # return the most recent response even if it is invalid

def process_images_with_gemini(image_batch, max_retries=3):
    """Run the Gemini model on a batch of images."""
    last_result = ""  # Variable holding the most recent response
    
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            
            # Include the list of image filenames first
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
            last_result = result  # Save the result
            
            # Print the response
            print("\n----- Gemini response (attempt {}/{}) -----".format(attempt+1, max_retries))
            print(result)
            print("----------------------------")
            
            # Validate the response: confirm a judgment is present for each file
            valid_response = True
            missing_files = []
            for img_path in image_batch:
                filename = os.path.basename(img_path)
                if filename not in result and f"filename: {filename}" not in result:
                    valid_response = False
                    missing_files.append(filename)
            
            if not valid_response:
                print(f"No judgment found in the Gemini response for the following files: {', '.join(missing_files)}. Retrying... (attempt {attempt+1}/{max_retries})")
            else:
                return result
            
            # Wait briefly, then retry
            time.sleep(2)
            
        except Exception as e:
            print(f"Gemini error: {e}. Retrying... (attempt {attempt+1}/{max_retries})")
            time.sleep(2)
    
    # Maximum number of retries exceeded
    print(f"Gemini failed: exceeded the maximum number of retries ({max_retries}).")
    return last_result  # return the most recent response even if it is invalid

def parse_results(text, filenames):
    """Parse filenames and judgments from a model response (supports several formats)."""
    results = {}
    base_filenames = [os.path.basename(f) for f in filenames]
    
    if not text or text.strip() == "":
        print("The response is empty.")
        return results
    
    print("\n=== Text being parsed ===")
    print(text)
    print("========================\n")
    
    # 1. Standard format (filename: xxx judgment: O/X)
    lines = text.strip().split('\n')
    for line in lines:
        if 'judgment:' in line.lower() or 'judgment :' in line.lower():
            parts = line.split('judgment:') if 'judgment:' in line.lower() else line.split('judgment :')
            filename_part = parts[0].strip()
            judgment_part = parts[1].strip() if len(parts) > 1 else ""
            
            # Extract the filename
            filename = ""
            if "filename:" in filename_part.lower():
                filename = filename_part.split("filename:")[1].strip()
            elif "filename :" in filename_part.lower():
                filename = filename_part.split("filename :")[1].strip()
            else:
                # If the filename is not clearly marked, match in order
                for f in base_filenames:
                    if f in filename_part:
                        filename = f
                        break
            
            # Strip brackets and quotes from the extracted filename
            filename = filename.strip("[]'\"\t ")
            
            # Extract the judgment (O or X)
            judgment = ""
            if "O" in judgment_part or "o" in judgment_part:
                judgment = "O"
            elif "X" in judgment_part or "x" in judgment_part:
                judgment = "X"
            
            if filename and judgment:
                results[filename] = judgment
                print(f"Parsed (standard format): {filename} -> {judgment}")
    
    # 2. Simple format (filename.png: O/X)
    if len(results) < len(base_filenames):
        for line in lines:
            if ':' in line and ('O' in line or 'X' in line or 'o' in line or 'x' in line):
                parts = line.split(':')
                if len(parts) >= 2:
                    filename_part = parts[0].strip()
                    judgment_part = parts[1].strip()
                    
                    # Match the filename
                    matched_filename = ""
                    for f in base_filenames:
                        if f in filename_part or filename_part in f:
                            matched_filename = f
                            break
                    
                    if not matched_filename:
                        continue
                    
                    # Extract the judgment
                    judgment = ""
                    if "O" in judgment_part or "o" in judgment_part:
                        judgment = "O"
                    elif "X" in judgment_part or "x" in judgment_part:
                        judgment = "X"
                    
                    if matched_filename and judgment and matched_filename not in results:
                        results[matched_filename] = judgment
                        print(f"Parsed (simple format): {matched_filename} -> {judgment}")
    
    # 3. Table format (markdown tables or other layouts)
    if len(results) < len(base_filenames):
        for line in lines:
            if '|' in line:
                parts = [part.strip() for part in line.split('|')]
                for part in parts:
                    if not part:
                        continue
                    
                    # Find the filename and judgment in each cell
                    file_match = None
                    for f in base_filenames:
                        if f in part:
                            file_match = f
                            break
                    
                    if not file_match:
                        continue
                    
                    # Extract the judgment
                    judgment = ""
                    if "O" in part or "o" in part:
                        judgment = "O"
                    elif "X" in part or "x" in part:
                        judgment = "X"
                    
                    if file_match and judgment and file_match not in results:
                        results[file_match] = judgment
                        print(f"Parsed (table format): {file_match} -> {judgment}")
    
    # 4. List format (filename and judgment near each other)
    if len(results) < len(base_filenames):
        for i, line in enumerate(lines):
            if not line.strip():
                continue
                
            # Find the filename on the current line
            file_match = None
            for f in base_filenames:
                if f in line:
                    file_match = f
                    break
            
            if not file_match or file_match in results:
                continue
            
            # Look for the judgment on the current or next line
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
                print(f"Parsed (list format): {file_match} -> {judgment}")
    
    # 5. Last resort: infer a judgment from the full text for each remaining file
    if len(results) < len(base_filenames):
        for f in base_filenames:
            if f not in results:
                # Find the surrounding context for the filename
                context = ""
                for line in lines:
                    if f in line:
                        context = line
                        break
                
                if not context:
                    continue
                
                # Look for O/X near the filename
                judgment = ""
                if "O" in context or "o" in context:
                    judgment = "O"
                elif "X" in context or "x" in context:
                    judgment = "X"
                
                if judgment:
                    results[f] = judgment
                    print(f"Parsed (context inference): {f} -> {judgment}")
    
    # 6. A different special format (filename.png: X)
    if len(results) < len(base_filenames):
        for line in lines:
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    filename_part = parts[0].strip()
                    value_part = parts[1].strip()
                    
                    # Check the filename
                    if filename_part.endswith('.png') or filename_part.endswith('.jpg') or filename_part.endswith('.jpeg'):
                        for f in base_filenames:
                            if filename_part == f:
                                # Extract the judgment
                                judgment = ""
                                if value_part.strip() == 'O' or value_part.strip() == 'o':
                                    judgment = "O"
                                elif value_part.strip() == 'X' or value_part.strip() == 'x':
                                    judgment = "X"
                                
                                if judgment and f not in results:
                                    results[f] = judgment
                                    print(f"Parsed (special format): {f} -> {judgment}")
    
    print(f"Final parse result ({len(results)}/{len(base_filenames)} files processed): {results}")
    return results

def create_log_directory():
    """Create the log directory."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def create_log_files(timestamp):
    """Create the log directory and the combined log file."""
    log_dir = create_log_directory()
    
    # Create a single combined log file
    combined_log_file = f"{log_dir}/combined_log_{timestamp}.log"
    
    with open(combined_log_file, 'w', encoding='utf-8') as f:
        f.write(f"=== Illegal online gambling site detection log ({timestamp}) ===\n\n")
    
    return log_dir, combined_log_file

def append_to_log(log_file, batch_number, gpt_result, claude_result, gemini_result, gpt_parsed, claude_parsed, gemini_parsed, current_filenames):
    """Append batch results to the log file."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    with open(log_file, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n\n{'='*50}\n")
        log_file.write(f"=== Batch {batch_number} log ({timestamp}) ===\n")
        log_file.write(f"{'='*50}\n\n")
        log_file.write(f"Processed images: {', '.join(current_filenames)}\n\n")
        
        log_file.write("=== GPT model response ===\n")
        log_file.write(gpt_result if gpt_result else "No response")
        log_file.write("\n\n")
        
        log_file.write("=== Claude model response ===\n")
        log_file.write(claude_result if claude_result else "No response")
        log_file.write("\n\n")
        
        log_file.write("=== Gemini model response ===\n")
        log_file.write(gemini_result if gemini_result else "No response")
        log_file.write("\n\n")
        
        log_file.write("=== Parsed result summary ===\n")
        log_file.write("Filename, GPT, Claude, Gemini\n")
        for filename in current_filenames:
            gpt_judgment = gpt_parsed.get(filename, "")
            claude_judgment = claude_parsed.get(filename, "")
            gemini_judgment = gemini_parsed.get(filename, "")
            log_file.write(f"{filename}, {gpt_judgment}, {claude_judgment}, {gemini_judgment}\n")
        
        # Check whether any responses were not parsed
        missing_gpt = [f for f in current_filenames if f not in gpt_parsed]
        missing_claude = [f for f in current_filenames if f not in claude_parsed]
        missing_gemini = [f for f in current_filenames if f not in gemini_parsed]
        
        if missing_gpt or missing_claude or missing_gemini:
            log_file.write("\n=== Missing judgments ===\n")
            if missing_gpt:
                log_file.write(f"Files missing from GPT: {', '.join(missing_gpt)}\n")
            if missing_claude:
                log_file.write(f"Files missing from Claude: {', '.join(missing_claude)}\n")
            if missing_gemini:
                log_file.write(f"Files missing from Gemini: {', '.join(missing_gemini)}\n")
    
    print(f"Appended results to the combined log file.")

def append_error_to_log(log_file, batch_number, error, current_filenames):
    """Append error information to the log file."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n\n{'='*50}\n")
        f.write(f"=== Batch {batch_number} error log ({timestamp}) ===\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Error while processing batch {batch_number}: {error}\n")
        f.write(f"Processed images: {', '.join(current_filenames)}\n")
    
    print(f"Error information appended to the combined log file.")

def main():
    # Generate a timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_csv = f"gambling_results_{timestamp}.csv"  # Append a timestamp to the filename
    
    # Create the log directory and files
    log_dir, combined_log_file = create_log_files(timestamp)
    
    # Get all image files
    all_images = get_all_image_files()
    total_images = len(all_images)
    
    if total_images == 0:
        print(f"No images found in the '{IMAGE_FOLDER}' folder.")
        return
    
    print(f"{total_images} images are available for processing.")
    
    # Ask for the starting index
    start_index_input = input(f"Which image number should processing start from? (1-{total_images}, default: 1): ").strip()
    start_index = 1
    
    try:
        if start_index_input:
            start_index = int(start_index_input)
            if start_index < 1:
                print("Values below 1 are not allowed. Starting from 1.")
                start_index = 1
            elif start_index > total_images:
                print(f"Values greater than {total_images} are not allowed. Starting from 1.")
                start_index = 1
    except ValueError:
        print("Not a valid number. Starting from 1.")
        start_index = 1
    
    # Convert the starting index into a batch index
    batch_start_index = ((start_index - 1) // BATCH_SIZE) * BATCH_SIZE
    
    # If some images were already processed, report it
    if start_index > 1:
        print(f"Starting processing from image {start_index}. (Images 1-{start_index-1} are skipped.)")
        
        # Check for an existing CSV file
        continue_csv = input("If an existing CSV file is available, continue appending to it? (y/n, default: n): ").strip().lower()
        
        if continue_csv == 'y':
            existing_csv = input("Enter the path to the existing CSV file: ").strip()
            if os.path.exists(existing_csv):
                output_csv = existing_csv
                print(f"Appending results to '{output_csv}'.")
            else:
                print(f"'{existing_csv}' not found. Saving results to a new file '{output_csv}'.")
                # Initialize the CSV file
                with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['filename', 'gpt', 'claude', 'gemini'])
        else:
            # Initialize the CSV file
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['filename', 'gpt', 'claude', 'gemini'])
    else:
        # Initialize the CSV file
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['filename', 'gpt', 'claude', 'gemini'])
    
    print(f"Results will be saved to '{output_csv}'.")
    print(f"Logs will be saved to '{combined_log_file}'.")
    
    # Process in batches
    batch_index = batch_start_index
    while batch_index < total_images:
        # Select the images for the current batch
        end_index = min(batch_index + BATCH_SIZE, total_images)
        current_batch = all_images[batch_index:end_index]
        current_filenames = [os.path.basename(img) for img in current_batch]
        batch_number = batch_index//BATCH_SIZE + 1
        
        print(f"\nProcessing batch {batch_number}... ({batch_index+1}-{end_index}/{total_images})")
        print("Images currently being processed:", ", ".join(current_filenames))
        
        try:
            # Run the GPT model
            print("\n===== Running the GPT model... =====")
            gpt_result = ""
            gpt_retry_count = 0
            max_model_retries = 3
            
            while gpt_retry_count < max_model_retries:
                try:
                    gpt_result = process_images_with_gpt(current_batch)
                    
                    # Parse the result (do not print the response again)
                    print("\n===== Parsing GPT results... =====")
                    gpt_parsed = parse_results(gpt_result, current_batch)
                    
                    # Validate the result
                    if len(gpt_parsed) == len(current_batch):
                        break  # stop once every file has a result
                    else:
                        gpt_retry_count += 1
                        if gpt_retry_count < max_model_retries:
                            print(f"GPT result is incomplete. Retrying GPT only... (attempt {gpt_retry_count}/{max_model_retries})")
                            time.sleep(2)
                        else:
                            print(f"Exceeded the maximum number of GPT retries ({max_model_retries}). Proceeding with the results obtained so far.")
                
                except Exception as e:
                    gpt_retry_count += 1
                    print(f"Error while processing GPT: {e}")
                    if gpt_retry_count < max_model_retries:
                        print(f"Retrying GPT... (attempt {gpt_retry_count}/{max_model_retries})")
                        time.sleep(2)
                    else:
                        print(f"Exceeded the maximum number of GPT retries ({max_model_retries}).")
                        gpt_parsed = {}  # set an empty result
            
            # Run the Claude model
            print("\n===== Running the Claude model... =====")
            claude_result = ""
            claude_retry_count = 0
            
            while claude_retry_count < max_model_retries:
                try:
                    claude_result = process_images_with_claude(current_batch)
                    
                    # Parse the result
                    print("\n===== Parsing Claude results... =====")
                    claude_parsed = parse_results(claude_result, current_batch)
                    
                    # Validate the result
                    if len(claude_parsed) == len(current_batch):
                        break  # stop once every file has a result
                    else:
                        claude_retry_count += 1
                        if claude_retry_count < max_model_retries:
                            print(f"Claude result is incomplete. Retrying Claude only... (attempt {claude_retry_count}/{max_model_retries})")
                            time.sleep(2)
                        else:
                            print(f"Exceeded the maximum number of Claude retries ({max_model_retries}). Proceeding with the results obtained so far.")
                
                except Exception as e:
                    claude_retry_count += 1
                    print(f"Error while processing Claude: {e}")
                    if claude_retry_count < max_model_retries:
                        print(f"Retrying Claude... (attempt {claude_retry_count}/{max_model_retries})")
                        time.sleep(2)
                    else:
                        print(f"Exceeded the maximum number of Claude retries ({max_model_retries}).")
                        claude_parsed = {}  # set an empty result
            
            # Run the Gemini model
            print("\n===== Running the Gemini model... =====")
            gemini_result = ""
            gemini_retry_count = 0
            
            while gemini_retry_count < max_model_retries:
                try:
                    gemini_result = process_images_with_gemini(current_batch)
                    
                    # Parse the result
                    print("\n===== Parsing Gemini results... =====")
                    gemini_parsed = parse_results(gemini_result, current_batch)
                    
                    # Validate the result
                    if len(gemini_parsed) == len(current_batch):
                        break  # stop once every file has a result
                    else:
                        gemini_retry_count += 1
                        if gemini_retry_count < max_model_retries:
                            print(f"Gemini result is incomplete. Retrying Gemini only... (attempt {gemini_retry_count}/{max_model_retries})")
                            time.sleep(2)
                        else:
                            print(f"Exceeded the maximum number of Gemini retries ({max_model_retries}). Proceeding with the results obtained so far.")
                
                except Exception as e:
                    gemini_retry_count += 1
                    print(f"Error while processing Gemini: {e}")
                    if gemini_retry_count < max_model_retries:
                        print(f"Retrying Gemini... (attempt {gemini_retry_count}/{max_model_retries})")
                        time.sleep(2)
                    else:
                        print(f"Exceeded the maximum number of Gemini retries ({max_model_retries}).")
                        gemini_parsed = {}  # set an empty result
            
            # Append the results to the log file
            append_to_log(
                combined_log_file, batch_number, gpt_result, claude_result, gemini_result,
                gpt_parsed, claude_parsed, gemini_parsed, current_filenames
            )
            
            # Append the results to the CSV file
            with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                
                for img_path in current_batch:
                    filename = os.path.basename(img_path)
                    gpt_judgment = gpt_parsed.get(filename, "")
                    claude_judgment = claude_parsed.get(filename, "")
                    gemini_judgment = gemini_parsed.get(filename, "")
                    
                    csv_writer.writerow([filename, gpt_judgment, claude_judgment, gemini_judgment])
                    print(f"Added to CSV: {filename}, GPT: {gpt_judgment}, Claude: {claude_judgment}, Gemini: {gemini_judgment}")
            
            print(f"Batch {batch_number} results saved to the CSV file.")
                
        except Exception as e:
            print(f"Unexpected error during processing: {e}")
            # Write the error to the log
            append_error_to_log(combined_log_file, batch_number, e, current_filenames)
            
            # Continue even after an unexpected error (no user input required)
            print(f"An error occurred, but continuing with the next batch.")
        
        # Move on to the next batch (automatically, no confirmation)
        batch_index = end_index
    
    print("\nAll processing complete.")
    print(f"Results saved to '{output_csv}'.")
    print(f"Detailed logs saved to '{combined_log_file}'.")

if __name__ == "__main__":
    main()
