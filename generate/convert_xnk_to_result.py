#!/usr/bin/env python3
"""
Convert xnk JSONL format to result.jsonl format.

Usage:
  python scripts/convert_xnk_to_result.py --input <xnk_jsonl> --output <result_json>

This script reads each line from the input JSONL (xnk) and produces a list of
items matching the example `result.jsonl` structure. It extracts the nearest
contents for tags <本轮患者发言>, <对话历史>, <就医档案> from the `input`
text, and maps doctor/think/answer fields from the `output`/`doctor` values.
"""
import argparse
import json
import re
from typing import Optional, Dict, Any, List


TAG_PATTERNS = {
    'patient_say': re.compile(r'<本轮患者发言>(.*?)</本轮患者发言>', re.S),
    'history': re.compile(r'<对话历史>(.*?)</对话历史>', re.S),
    'report': re.compile(r'<就医档案>(.*?)</就医档案>', re.S),
}


def extract_tag_nearest(text: str, tag_re: re.Pattern, start_pos: int = 0) -> Optional[str]:
    """Return the nearest (last) match for the tag in the text after start_pos, stripped, or None.

    If start_pos > 0, only consider matches that start at or after that index. This helps
    avoid picking up template/task descriptions that appear before the real content.
    """
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
    """Given the `output` field from xnk entry, try to extract think/answer.

    The xnk sample shows `output` as a string containing <think>...</think><answer>...</answer>
    but some entries may have a `doctor` field already (string) that includes tags.
    We'll attempt to extract from either `output` or `doctor` keys.
    """
    think = None
    answer = None
    if not output_field:
        return {'think': None, 'answer': None}

    if isinstance(output_field, str):
        text = output_field
    elif isinstance(output_field, dict):
        # sometimes output might already be a dict with keys
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
    """Return an index in text after which real content likely starts.

    Many xnk inputs contain a long <任务>... block at the top. We prefer parsing tags
    that occur after the closing </任务> if present. Otherwise return 0.
    """
    if not text:
        return 0
    # find end of task block
    m = re.search(r'</任务>', text)
    if m:
        return m.end()
    # fallback: if there is a '<本轮患者发言>' later, prefer content after first line break
    return 0


def convert_entry(xnk_obj: Dict[str, Any]) -> Dict[str, Any]:
    # default skeleton for result item based on example
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

    # try to copy department from instruction or elsewhere if present
    # primary text to parse tags is in xnk_obj.get('input', '')
    raw_input = xnk_obj.get('input')
    if isinstance(raw_input, dict):
        # if input is already structured, join likely text fields
        combined_input_text = ''
        for v in raw_input.values():
            if isinstance(v, str):
                combined_input_text += '\n' + v
    else:
        combined_input_text = raw_input or ''

    # determine a content start position to avoid template blocks
    content_start = _find_content_region(combined_input_text)

    # extract tags
    # prefer the last match that occurs after the content_start (to skip template blocks);
    # if none found after content_start, fallback to the last match anywhere in the text
    for key, pattern in TAG_PATTERNS.items():
        found_after = find_last_match_after(combined_input_text, pattern, after_pos=content_start)
        if found_after:
            val = found_after[0]
        else:
            # fallback to the last match anywhere
            all_matches = [m.group(1).strip() for m in pattern.finditer(combined_input_text)]
            val = all_matches[-1] if all_matches else None
        if key == 'patient_say':
            # Strict: only accept explicit <本轮患者发言> matches. No fallback.
            if val is None:
                result_item['input']['patient_say'] = None
            else:
                v = val.strip()
                result_item['input']['patient_say'] = None if v == '无' or v == '' else v
        elif key == 'history':
            # Only accept explicit history matches; if not present or is '无', return empty list
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
            # Accept explicit report matches; otherwise keep empty dict
            if val is None:
                result_item['input']['report'] = {}
            else:
                v = val.strip()
                if v == '无' or v == '':
                    result_item['input']['report'] = {}
                else:
                    result_item['input']['report'] = {'text': v}

    # No fallback: if patient_say is None, keep it None (strict mapping)

    # department: try to detect cardiology from instruction or existing fields
    dept = None
    if isinstance(xnk_obj.get('instruction'), str) and '心内科' in xnk_obj.get('instruction'):
        dept = '心内科'
    # if input contains department mention
    if isinstance(combined_input_text, str) and '心内科' in combined_input_text:
        dept = '心内科'
    result_item['input']['department'] = dept

    # doctor: prefer <answer> from `output` field (per spec). If absent, try `doctor` field.
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
            # if doctor is a plain string without tags, use it whole
            if isinstance(xnk_obj['doctor'], str):
                doctor_text = xnk_obj['doctor'].strip()
            else:
                doctor_text = json.dumps(xnk_obj['doctor'], ensure_ascii=False)

    result_item['doctor'] = doctor_text

    # fill output_raw.llm think/response strictly from the 'doctor' field's tags only.
    # Do NOT fallback to 'output' field.
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
    results: List[Dict[str, Any]] = []
    with open(input_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # try to skip or continue
                print('Warning: skipping invalid json line')
                continue
            converted = convert_entry(obj)
            results.append(converted)

    # save as a JSON array (like the example)
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Path to xnk jsonl input')
    parser.add_argument('--output', '-o', required=True, help='Path to result json output')
    args = parser.parse_args()
    convert_file(args.input, args.output)


if __name__ == '__main__':
    main()
