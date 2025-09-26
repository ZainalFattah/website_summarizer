import os
import glob
import json
import re
from pathlib import Path

from dotenv import load_dotenv

# --- Import Docling ---
from docling.document_converter import DocumentConverter, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import ConversionStatus

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ----------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------
os.environ["DOCLING_ACCELERATOR"] = "cuda"
load_dotenv()

# Folder input PDF
all_file_paths = glob.glob("./data_pdf/*.pdf")
print(f"[INFO] Ditemukan {len(all_file_paths)} total file PDF di ./data_pdf/")

# Filter file berdasarkan ukuran (untuk skip corrupt file)
MIN_SIZE_KB = 20
MIN_SIZE_BYTES = MIN_SIZE_KB * 1024

FILE_PATHS = []
skipped_files = []

for path in all_file_paths:
    if os.path.getsize(path) > MIN_SIZE_BYTES:
        FILE_PATHS.append(path)
    else:
        skipped_files.append(os.path.basename(path))

print(f"[INFO] Akan memproses {len(FILE_PATHS)} file (ukuran > {MIN_SIZE_KB} KB).")
if skipped_files:
    print(f"[WARNING] Melewati {len(skipped_files)} file karena diduga corrupt: {', '.join(skipped_files)}")

# Model embedding
EMBED_MODEL_ID = "BAAI/bge-m3"
EXPORT_TYPE = "html"

# Konfigurasi chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Database vektor
CHROMA_DIR = "./db_chroma"
COLLECTION_NAME = "main_research_papers"


# ----------------------------------------------------------------------
# 1. Load dokumen sebagai HTML dengan Docling
# ----------------------------------------------------------------------
if not FILE_PATHS:
    print("[INFO] Tidak ada file valid untuk diproses. Program berhenti.")
    exit()

# Buat converter (tanpa backend, default sudah cukup)
converter = DocumentConverter()

processed_results = []
for file_path in FILE_PATHS:
    try:
        conv_result = converter.convert(file_path)
        if conv_result.status == ConversionStatus.SUCCESS:
            html_content = conv_result.document.export_to_html()
            processed_results.append({
                "content": html_content,
                "metadata": {
                    "source": file_path,
                    "title": Path(file_path).stem
                }
            })
            print(f"[INFO] Berhasil convert: {os.path.basename(file_path)}")
        else:
            print(f"[ERROR] Gagal convert {file_path}, status: {conv_result.status}")
    except Exception as e:
        print(f"[ERROR] Exception saat convert {file_path}: {e}")

print(f"[INFO] Docling berhasil memproses {len(processed_results)} dokumen sebagai HTML.")


# ----------------------------------------------------------------------
# 2. Ekstrak teks relevan & chunking
# ----------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

START_PATTERNS = r'\b(abstract|abstrak)\b'
END_PATTERNS = r'\b(conclusion|conclusions|kesimpulan|references|daftar pustaka|acknowledgements)\b'

final_splits = []
for result in processed_results:
    html_content = result.get("content", "")
    metadata = result.get("metadata", {})
    source_path = metadata.get("source", "Unknown File")
    title = metadata.get("title", os.path.basename(source_path))

    start_match = re.search(START_PATTERNS, html_content, re.IGNORECASE)

    if start_match:
        text_after_abstract = html_content[start_match.start():]
        end_match = re.search(END_PATTERNS, text_after_abstract, re.IGNORECASE)

        if end_match:
            relevant_html = text_after_abstract[:end_match.end()]
            clean_text = re.sub("<[^<]+?>", " ", relevant_html)
            clean_text = " ".join(clean_text.split())

            chunks = text_splitter.split_text(clean_text)

            for chunk_content in chunks:
                new_doc = Document(
                    page_content=chunk_content,
                    metadata={"title": title, "source": source_path}
                )
                final_splits.append(new_doc)
            print(f"[INFO] Memproses '{title}', menghasilkan {len(chunks)} chunk.")
        else:
            print(f"[WARNING] 'Abstract' ditemukan tapi tidak ada 'Conclusion/References' di file: {source_path}")
    else:
        print(f"[WARNING] Tidak ditemukan 'Abstract' di file: {source_path}")

print(f"\n[INFO] Total chunks berhasil dibuat: {len(final_splits)}")


# ----------------------------------------------------------------------
# 3. Print contoh hasil chunking
# ----------------------------------------------------------------------
print("\n=== Contoh hasil chunking ===")
for i, doc in enumerate(final_splits[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print("Konten :", doc.page_content[:300].replace("\n", " "), "...")
    print("Metadata:", json.dumps(doc.metadata, indent=2))


# ----------------------------------------------------------------------
# 4. Buat embedding & simpan ke Chroma
# ----------------------------------------------------------------------
if final_splits:
    embedding = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        model_kwargs={"device": "cuda"},
        encode_kwargs={
            "normalize_embeddings": True,
        }
    )

    vectorstore = Chroma.from_documents(
        documents=final_splits,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    vectorstore.persist()
    print(f"\n[INFO] Embedding berhasil disimpan di {CHROMA_DIR}")
else:
    print("\n[INFO] Tidak ada chunk yang dibuat, embedding dilewati.")
