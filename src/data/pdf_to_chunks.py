import os
import re
import json
import tqdm

try:
    import pdfplumber
except Exception as e:
    raise ImportError(
        "pdfplumber is required to run this script. Install with: pip install pdfplumber"
    )

from langchain.text_splitter import RecursiveCharacterTextSplitter


def ends_with_ending_punctuation(s):
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)


def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    else:
        return title.strip() + ". " + content.strip()


def extract_text_from_pdf(fpath):
    """Extracts text from a PDF file using pdfplumber."""
    pages = []
    with pdfplumber.open(fpath) as pdf:
        for page in pdf.pages:
            try:
                txt = page.extract_text()
            except Exception:
                txt = None
            if txt:
                pages.append(txt)
    return "\n".join(pages).strip()


if __name__ == "__main__":
    # Input directory containing PDFs
    fdir = "./corpus/pdf"

    if not os.path.exists(fdir):
        raise SystemExit(f"Input directory not found: {fdir}. Please put your PDFs under this folder.")

    out_dir = os.path.join("corpus", "pdf", "chunk")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fnames = sorted([fname for fname in os.listdir(fdir) if fname.lower().endswith('.pdf')])

    if len(fnames) == 0:
        print(f"No PDF files found in {fdir}. Nothing to do.")
        raise SystemExit(0)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for fname in tqdm.tqdm(fnames):
        fpath = os.path.join(fdir, fname)
        outpath = os.path.join(out_dir, fname.replace('.pdf', '.jsonl'))

        # Skip if already processed
        if os.path.exists(outpath):
            continue

        full_text = extract_text_from_pdf(fpath)
        if not full_text:
            # skip empty pdfs
            continue

        # normalize whitespace
        full_text = re.sub(r"\s+", " ", full_text).strip()

        texts = text_splitter.split_text(full_text)
        title = fname.replace('.pdf', '')

        saved_text = [
            json.dumps({
                "id": "_".join([title, str(i)]),
                "title": title,
                "content": texts[i],
                "contents": concat(title, texts[i]),
            }, ensure_ascii=False)
            for i in range(len(texts))
        ]

        if len(saved_text) > 0:
            with open(outpath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(saved_text))
