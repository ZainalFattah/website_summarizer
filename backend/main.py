import os
import re
import json
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
from docling.document_converter import DocumentConverter
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

# ======================================================================================
# 1. INITIAL SETUP & MODEL LOADING
# ======================================================================================
LLM = None
DOC_CONVERTER = None
CHROMA_CLIENT = None
EMBEDDING_MODEL = None

print("✅ Server starting...")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global LLM, DOC_CONVERTER, CHROMA_CLIENT, EMBEDDING_MODEL
    print("⏳ Loading Document Converter...")
    os.environ["DOCLING_ACCELERATOR"] = "cuda"
    DOC_CONVERTER = DocumentConverter()
    print("✅ Document Converter loaded.")

    print("⏳ Loading Embedding Model BAAI/bge-m3...")
    EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cuda"}, encode_kwargs={"normalize_embeddings": True})
    print("✅ Embedding Model loaded.")
    
    print("⏳ Initializing ChromaDB Client...")
    CHROMA_CLIENT = chromadb.PersistentClient(path="./BGEM3_V2")
    
    try:
        # --- PERUBAHAN UTAMA DI SINI ---
        # Menggunakan get_or_create_collection untuk mencegah error.
        print("🔎 Accessing or creating main collection 'main_research_papers'...")
        main_collection = CHROMA_CLIENT.get_or_create_collection("main_research_papers")
        
        # Jika koleksi baru saja dibuat (kosong), tambahkan data dummy
        # agar tidak error saat di-query untuk pertama kali.
        if main_collection.count() == 0:
            print("Collection was empty. Populating with a dummy entry...")
            main_collection.add(
                documents=["Artificial intelligence is a branch of computer science."],
                ids=["dummy_entry_1"]
            )

        print(f"✅ Main collection ready with {main_collection.count()} documents.")
    except Exception as e:
        # Blok ini sekarang hanya akan berjalan jika ada error yang benar-benar serius
        print(f"❌ CRITICAL ERROR: Could not load or create main collection. Error: {e}")

    print("✅ ChromaDB Client initialized.")

    model_path = "./models/phi-4-Q4_K_M.gguf"
    print(f"⏳ Loading GGUF model from: {model_path}...")
    LLM = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=25, temperature=0.5, top_p=0.5, verbose=True)
    print("✅ GGUF model loaded successfully.")
    
    print("🚀 Server ready!")
    yield
    print("🛑 Server shutting down...")


app = FastAPI(title="RAG Summarization API", version="8.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================================================
# 2. HELPER FUNCTIONS & PROMPTS
# ======================================================================================

def convert_pdf_to_markdown(pdf_path: str) -> str:
    result = DOC_CONVERTER.convert(pdf_path)
    return result.document.export_to_markdown()

def clean_text(text: str) -> str:
    text = re.sub(r'\|.*\|', '', text); text = re.sub(r'\[\d+\]', '', text); text = re.sub(r'#+\s.*', '', text); text = re.sub(r'\n\s*\n', '\n', text); return text.strip()

def chunk_text_by_char(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    if len(text) <= chunk_size: return [text]
    chunks = []; start = 0
    while start < len(text):
        end = start + chunk_size; chunks.append(text[start:end]); start += chunk_size - overlap
    return chunks

PROMPT_TEMPLATES = {
    "en": {
        "map": {"system": "Summarize the key points of the following text in a single, well-written paragraph in English.", "user": "Please summarize this text: {chunk}"},
        "reduce": {"system": "You are an expert academic research summarizer. Generate a final structured summary in a valid JSON format in English. The JSON keys must be: 'research_objective', 'methods', 'main_results', 'conclusions'. If a key's information is not found, return an empty string.", "user": "Here are the summaries:\n\n---\n{combined_summary}\n---"},
        "qa": {"system": "You are a helpful Q&A assistant. Based on the provided context, answer the user's question **in the same language they used for the question**. If the context isn't relevant or doesn't contain the answer, state that you cannot answer based on the information.", "user": "Context:\n---\n{context}\n---\nQuestion: {question}"}
    },
    "id": {
        "map": {"system": "Ringkaslah poin-poin utama dari teks berikut dalam satu paragraf yang ditulis dengan baik dalam Bahasa Indonesia.", "user": "Tolong ringkas teks ini: {chunk}"},
        "reduce": {"system": "Anda adalah seorang ahli peringkas riset akademis. Buatlah sebuah ringkasan akhir terstruktur dalam format JSON yang valid dalam Bahasa Indonesia. Kunci JSON harus: 'research_objective', 'methods', 'main_results', 'conclusions'. Jika informasi untuk kunci tidak ditemukan, kembalikan string kosong.", "user": "Berikut adalah kumpulan ringkasan:\n\n---\n{combined_summary}\n---"},
        "qa": {"system": "Anda adalah asisten Tanya Jawab yang membantu. Berdasarkan konteks yang diberikan, jawab pertanyaan pengguna **dalam bahasa yang sama dengan yang mereka gunakan untuk bertanya**. Jika konteks tidak relevan atau tidak berisi jawaban, nyatakan bahwa Anda tidak dapat menjawab berdasarkan informasi yang diberikan.", "user": "Konteks:\n---\n{context}\n---\nPertanyaan: {question}"}
    }
}

# ======================================================================================
# 3. Pydantic Models for API
# ======================================================================================

class QARequest(BaseModel):
    document_id: Optional[str] = None
    question: str
    lang: str = "en"

# ======================================================================================
# 4. API ENDPOINTS
# ======================================================================================

@app.get("/")
def root(): return {"message": "RAG Summarization API is running."}

@app.post("/summarize")
async def summarize_pdf(file: UploadFile = File(...), lang: str = Form("en")):
    if not file.filename.endswith(".pdf"): raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
    if lang not in PROMPT_TEMPLATES: lang = "en"
    tmp_pdf_path = None
    document_id = str(uuid.uuid4())
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read()); tmp_pdf_path = tmp.name
        markdown_text = convert_pdf_to_markdown(tmp_pdf_path)
        cleaned_text = clean_text(markdown_text)
        if not cleaned_text or cleaned_text.isspace(): raise HTTPException(status_code=500, detail="Failed to extract meaningful text from PDF.")
        
        text_chunks = chunk_text_by_char(cleaned_text, chunk_size=3000, overlap=200)
        prompts = PROMPT_TEMPLATES[lang]
        chunk_summaries = []
        for i, chunk in enumerate(text_chunks):
            print(f"🔎 Summarizing chunk {i+1}/{len(text_chunks)}...")
            map_messages = [{"role": "system", "content": prompts['map']['system']}, {"role": "user", "content": prompts['map']['user'].format(chunk=chunk)}]
            response = LLM.create_chat_completion(messages=map_messages, max_tokens=512, temperature=0.2)
            chunk_summaries.append(response['choices'][0]['message']['content'])
        
        combined_summary = "\n\n".join(chunk_summaries)
        reduce_messages = [{"role": "system", "content": prompts['reduce']['system']}, {"role": "user", "content": prompts['reduce']['user'].format(combined_summary=combined_summary)}]
        final_response = LLM.create_chat_completion(messages=reduce_messages, max_tokens=1024, temperature=0.1, response_format={"type": "json_object"})
        llm_output_text = final_response['choices'][0]['message']['content']
        
        try:
            structured_summary = json.loads(llm_output_text)
            final_summary = { "Tujuan Penelitian": structured_summary.get("research_objective", "Tidak ditemukan."), "Metode": structured_summary.get("methods", "Tidak ditemukan."), "Hasil Utama": structured_summary.get("main_results", "Tidak ditemukan."), "Kesimpulan": structured_summary.get("conclusions", "Tidak ditemukan.") }
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(status_code=500, detail="Failed to parse summary from LLM.")
        
        print(f"⏳ Vectorizing document... Document ID: {document_id}")
        doc_chunks_for_qa = chunk_text_by_char(cleaned_text, chunk_size=1000, overlap=150)
        
        # Gunakan embedding model yang telah dimuat
        embeddings = EMBEDDING_MODEL.embed_documents(doc_chunks_for_qa)
        
        collection = CHROMA_CLIENT.create_collection(name=document_id)
        collection.add(embeddings=embeddings, documents=doc_chunks_for_qa, ids=[f"chunk_{i}" for i in range(len(doc_chunks_for_qa))])
        
        print(f"✅ Document vectorized and stored in '{document_id}'.")
        return {"filename": file.filename, "structured_summary": final_summary, "document_id": document_id}
    except Exception as e:
        print(f"❌ Error in /summarize: {e}")
        try: CHROMA_CLIENT.delete_collection(name=document_id)
        except Exception: pass
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        if tmp_pdf_path and os.path.exists(tmp_pdf_path): os.remove(tmp_pdf_path)

@app.post("/qa")
async def question_answering(request: QARequest):
    try:
        print(f"Received Q&A request. Document ID: {request.document_id}, Lang: {request.lang}")
        
        context_text = None
        found_relevant_context = False
        
        # Ubah query menjadi embedding
        query_embedding = EMBEDDING_MODEL.embed_query(request.question)

        # Langkah 1: Jika ada dokumen yang diunggah, cari di sana terlebih dahulu.
        if request.document_id:
            try:
                print(f"🔎 Step 1: Querying temporary collection '{request.document_id}'...")
                temp_collection = CHROMA_CLIENT.get_collection(name=request.document_id)
                results = temp_collection.query(query_embeddings=[query_embedding], n_results=2)
                
                if results and results['documents'] and results['distances'][0][0] < 1.0:
                    context_text = "\n\n".join(results['documents'][0])
                    found_relevant_context = True
                    print("✅ Found relevant context in the temporary document.")
                else:
                    print("⚠️ No relevant context in temp doc. Proceeding to fallback.")
            except Exception as e:
                print(f"⚠️ Could not query temp collection '{request.document_id}': {e}. Proceeding to fallback.")

        # Langkah 2: Jika tidak ada konteks relevan, cari di basis data utama.
        if not found_relevant_context:
            try:
                print("🔎 Step 2: Querying main collection 'main_research_papers'...")
                main_collection = CHROMA_CLIENT.get_collection("main_research_papers")
                results = main_collection.query(query_embeddings=[query_embedding], n_results=1)
                
                if results and results['documents'][0] and results['distances'][0][0] < 1.2:
                    context_text = "\n\n".join(results['documents'][0])
                    found_relevant_context = True
                    print("✅ Found relevant context in the main collection.")
                else:
                    print("⚠️ No relevant context found in the main collection.")
            except Exception as e:
                print(f"❌ Error querying main collection: {e}")

        # Langkah 3: Berdasarkan hasil, generate jawaban atau kembalikan pesan standar.
        if found_relevant_context:
            print("🧠 Generating answer based on found context...")
            prompts = PROMPT_TEMPLATES.get(request.lang, PROMPT_TEMPLATES["en"])
            qa_messages = [{"role": "system", "content": prompts['qa']['system']}, {"role": "user", "content": prompts['qa']['user'].format(context=context_text, question=request.question)}]
            response = LLM.create_chat_completion(messages=qa_messages, max_tokens=512, temperature=0.3)
            answer = response['choices'][0]['message']['content']
            return {"answer": answer}
        else:
            print("🤷 No relevant context found. Returning standard response.")
            NO_CONTEXT_MESSAGES = {
                "id": "Hmm, saya tidak melihat jawaban yang pas dari data saat ini. Bisa jadi pertanyaannya perlu sedikit diperjelas.",
                "en": "Hmm, I can't seem to find a suitable answer from the current data. Perhaps the question could be clarified a bit."
            }
            answer = NO_CONTEXT_MESSAGES.get(request.lang, NO_CONTEXT_MESSAGES["en"])
            return {"answer": answer}

    except Exception as e:
        print(f"❌ FATAL Error in /qa endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

