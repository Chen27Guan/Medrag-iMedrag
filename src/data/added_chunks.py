import os
import json
import re
import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter


def ends_with_ending_punctuation(s):
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)


def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    else:
        return title.strip() + ". " + content.strip()


def chunk_file(fpath, out_dir, chunk_size=1000, chunk_overlap=200):
    """Read a JSON file containing a list of objects with a 'text' field and
    write chunked jsonl file(s) to out_dir. If the input file is named foo.json,
    output will be out_dir/foo.jsonl where each line is a JSON object with keys:
    - id: <filename_without_ext>_<index>
    - title: filename without extension
    - content: the chunk text
    - contents: title + chunk (with punctuation fixed)
    """
    with open(fpath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Failed to load JSON from {fpath}: {e}")
            return

    if not isinstance(data, list):
        print(f"Expected a list at top level in {fpath}, got {type(data)}")
        return

    texts = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        text = item.get('text') or item.get('content') or item.get('contents')
        if not text:
            continue
        texts.append(re.sub(r"\s+", " ", text.strip()))

    if not texts:
        print(f"No text items found in {fpath}")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    fname = os.path.basename(fpath)
    name_no_ext = os.path.splitext(fname)[0]
    out_path = os.path.join(out_dir, name_no_ext + '.jsonl')

    all_chunks = []
    for i, t in enumerate(texts):
        chunks = splitter.split_text(t)
        for j, c in enumerate(chunks):
            obj = {
                'id': '_'.join([name_no_ext, f"{i}", f"{j}"]),
                'title': name_no_ext,
                'content': re.sub(r"\s+", " ", c),
                'contents': concat(name_no_ext, re.sub(r"\s+", " ", c))
            }
            all_chunks.append(json.dumps(obj, ensure_ascii=False))

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as out_f:
        out_f.write('\n'.join(all_chunks))


if __name__ == '__main__':
    fdir = '../../corpus/merge'
    out_dir = os.path.join(fdir, 'chunk')

    if not os.path.exists(fdir):
        print(f"Input directory {fdir} does not exist")
        raise SystemExit(1)

    fnames = sorted([p for p in os.listdir(fdir) if p.lower().endswith('.json')])

    if not fnames:
        print(f"No JSON files found in {fdir}")
        raise SystemExit(0)

    for fname in tqdm.tqdm(fnames):
        fpath = os.path.join(fdir, fname)
        chunk_file(fpath, out_dir)
