#!/usr/bin/env python3
"""
Convert xnk JSONL format to result.jsonl format.

Usage:
  # Single file mode:
  python convert_script.py --input <xnk.jsonl> --output <result.jsonl>
  
  # Directory batch mode:
  python convert_script.py --input <input_dir> --output <output_dir>

This script reads each line from the input JSONL (xnk) and produces a list of
items matching the example `result.jsonl` structure. It extracts the nearest
contents for tags <本轮患者发言>, <对话历史>, <就医档案> from the `input`
text, and maps doctor/think/answer fields from the `output`/`doctor` values.
"""
import argparse
import json
import re
import os
from typing import Optional, Dict, Any, List


TAG_PATTERNS = {
    'patient_say': re.compile(r'<本轮患者发言>(.*?)</本轮患者发言>', re.S),
    'history': re.compile(r'<对话历史>(.*?)</对话历史>', re.S),
    'report': re.compile(r'<就医档案>(.*?)</就医档案>', re.S),
}


def extract_tag_nearest(text: str, tag_re: re.Pattern, start_pos: int = 0) -> Optional[str]:
    """Return the nearest (last) match for the tag in the text after start_pos, stripped, or None."""
    if not text:
        return None
    matches = []
    for m in tag_re.finditer(text):
        if m.start() >= start_pos:
            matches.append(m.group(1))
    if not matches:
        return None
    val = matches[-1].strip()
    return val


def find_last_match_after(text: str, tag_re: re.Pattern, after_pos: int = 0):
    """Return tuple (matched_text, start_index) for the last tag match whose start >= after_pos, else None."""
    if not text:
        return None
    last = None
    for m in tag_re.finditer(text):
        if m.start() >= after_pos:
            last = (m.group(1).strip(), m.start(), m.end())
    return last


def extract_think_answer_from_output(output_field: Any) -> Dict[str, Optional[str]]:
    """Given the `output` field from xnk entry, try to extract think/answer."""
    think = None
    answer = None
    if not output_field:
        return {'think': None, 'answer': None}

    if isinstance(output_field, str):
        text = output_field
    elif isinstance(output_field, dict):
        text = json.dumps(output_field, ensure_ascii=False)
    else:
        text = str(output_field)

    th = re.search(r'<think>(.*?)</think>', text, re.S)
    an = re.search(r'<answer>(.*?)</answer>', text, re.S)
    if th:
        think = th.group(1).strip()
    if an:
        answer = an.group(1).strip()
    return {'think': think, 'answer': answer}


def _find_content_region(text: str) -> int:
    """Return an index in text after which real content likely starts."""
    if not text:
        return 0
    m = re.search(r'</任务>', text)
    if m:
        return m.end()
    return 0


def convert_entry(xnk_obj: Dict[str, Any]) -> Dict[str, Any]:
    # default skeleton for result item
    result_item: Dict[str, Any] = {
        'input': {
            'patient_say': None,
            'history': [],
            'tree': None,
            'tree_node': None,
            'node_flag': False,
            'report': {},
            'medical_orders': [],
            'department': None,
        },
        'doctor': None,
        'output_raw': {
            'state': 'success',
            'content': {
                'llm': {
                    'think': None,
                    'response': None
                }
            }
        }
    }

    raw_input = xnk_obj.get('input')
    if isinstance(raw_input, dict):
        combined_input_text = ''
        for v in raw_input.values():
            if isinstance(v, str):
                combined_input_text += '\n' + v
    else:
        combined_input_text = raw_input or ''

    content_start = _find_content_region(combined_input_text)

    for key, pattern in TAG_PATTERNS.items():
        found_after = find_last_match_after(combined_input_text, pattern, after_pos=content_start)
        if found_after:
            val = found_after[0]
        else:
            all_matches = [m.group(1).strip() for m in pattern.finditer(combined_input_text)]
            val = all_matches[-1] if all_matches else None
            
        if key == 'patient_say':
            if val is None:
                result_item['input']['patient_say'] = None
            else:
                v = val.strip()
                result_item['input']['patient_say'] = None if v == '无' or v == '' else v
        elif key == 'history':
            if val is None:
                result_item['input']['history'] = []
            else:
                v = val.strip()
                if v == '无' or v == '':
                    result_item['input']['history'] = []
                else:
                    lines = [ln.strip() for ln in v.splitlines() if ln.strip()]
                    result_item['input']['history'] = lines
        elif key == 'report':
            if val is None:
                result_item['input']['report'] = {}
            else:
                v = val.strip()
                if v == '无' or v == '':
                    result_item['input']['report'] = {}
                else:
                    result_item['input']['report'] = {'text': v}

    dept = None
    if isinstance(xnk_obj.get('instruction'), str) and '心内科' in xnk_obj.get('instruction'):
        dept = '心内科'
    if isinstance(combined_input_text, str) and '心内科' in combined_input_text:
        dept = '心内科'
    result_item['input']['department'] = dept

    doctor_text = None
    if 'output' in xnk_obj and xnk_obj['output']:
        oa_out = extract_think_answer_from_output(xnk_obj['output'])
        if oa_out.get('answer'):
            doctor_text = oa_out['answer']
    if doctor_text is None and 'doctor' in xnk_obj and xnk_obj['doctor']:
        oa_doc = extract_think_answer_from_output(xnk_obj['doctor'])
        if oa_doc.get('answer'):
            doctor_text = oa_doc['answer']
        else:
            if isinstance(xnk_obj['doctor'], str):
                doctor_text = xnk_obj['doctor'].strip()
            else:
                doctor_text = json.dumps(xnk_obj['doctor'], ensure_ascii=False)

    result_item['doctor'] = doctor_text

    think = None
    response = None
    if 'doctor' in xnk_obj and xnk_obj['doctor']:
        oa = extract_think_answer_from_output(xnk_obj['doctor'])
        think = oa.get('think')
        response = oa.get('answer')

    result_item['output_raw']['content']['llm']['think'] = think
    result_item['output_raw']['content']['llm']['response'] = response

    return result_item


def convert_file(input_path: str, output_path: str) -> None:
    """Reads a single file and writes to output_path."""
    print(f"Processing file: {input_path} -> {output_path}")
    results: List[Dict[str, Any]] = []
    
    try:
        with open(input_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    print(f'Warning: skipping invalid json line in {input_path}')
                    continue
                converted = convert_entry(obj)
                results.append(converted)

        with open(output_path, 'w', encoding='utf-8') as fout:
            json.dump(results, fout, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Path to xnk jsonl input file or directory')
    parser.add_argument('--output', '-o', required=True, help='Path to result json output file or directory')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    # Scenario 1: Input is a directory (Batch mode)
    if os.path.isdir(input_path):
        # Ensure output directory exists
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
            print(f"Created output directory: {output_path}")
        elif not os.path.isdir(output_path):
            print(f"Error: Input is a directory, but output path '{output_path}' is a file.")
            return

        # Iterate through all .jsonl files in the input directory
        files_found = 0
        for filename in os.listdir(input_path):
            if filename.endswith(".jsonl"):
                files_found += 1
                source_file = os.path.join(input_path, filename)
                
                # Construct new filename: name.jsonl -> name_result.jsonl
                file_root, file_ext = os.path.splitext(filename)
                new_filename = f"{file_root}_result{file_ext}"
                target_file = os.path.join(output_path, new_filename)
                
                convert_file(source_file, target_file)
        
        if files_found == 0:
            print(f"No .jsonl files found in {input_path}")
        else:
            print(f"Batch processing complete. Processed {files_found} files.")

    # Scenario 2: Input is a single file
    elif os.path.isfile(input_path):
        # If output path is an existing directory, put the file inside it
        if os.path.isdir(output_path):
             file_root, file_ext = os.path.splitext(os.path.basename(input_path))
             new_filename = f"{file_root}_result{file_ext}"
             output_path = os.path.join(output_path, new_filename)
        
        convert_file(input_path, output_path)
        print("Single file processing complete.")

    else:
        print(f"Error: Input path '{input_path}' does not exist.")


if __name__ == '__main__':
    main()