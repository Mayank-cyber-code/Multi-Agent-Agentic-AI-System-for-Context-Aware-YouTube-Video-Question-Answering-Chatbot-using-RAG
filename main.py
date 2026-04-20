import os
import tempfile
import re

import shutil
import traceback
from typing import TypedDict
import json  # 🔥 ADD THIS
from datetime import datetime

print("🔥 Starting FastAPI app...")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from math import sqrt
import requests
import wikipedia
import socket
socket.setdefaulttimeout(5)
from deep_translator import GoogleTranslator

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ===== 🔥 GLOBAL EMBEDDINGS CACHE =====
EMBEDDINGS = OpenAIEmbeddings()

from langgraph.graph import StateGraph, END
from langsmith import traceable

from google.cloud import storage

print("🚀 App starting...")

# ===== ENV =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")

# ===== INIT =====
def get_llm(task="qa"):
    if task == "summary":
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.3, timeout=30)
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.1, timeout=30)


def get_embeddings():
    return EMBEDDINGS

def get_bucket():
    if not GCS_BUCKET_NAME:
        raise Exception("GCS_BUCKET_NAME not set")
    return storage.Client().bucket(GCS_BUCKET_NAME)

def safe_llm_call(prompt, task="qa", retries=2):
    for i in range(retries):
        try:
            return get_llm(task).invoke(prompt).content
        except Exception as e:
            print(f"❌ LLM error (attempt {i+1}):", e)
    return "Error generating response"

def search_tavily(query):
    try:
        print("🌐 Tavily search running...")

        res = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": os.getenv("TAVILY_API_KEY"),
                "query": query,
                "search_depth": "basic"
            },
            timeout=10
        )

        data = res.json()

        results = [
            r.get("content", "")
            for r in data.get("results", [])
        ]

        return "\n\n".join(results)

    except Exception as e:
        print("❌ Tavily error:", e)
        return None

def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sqrt(sum(a * a for a in vec1))
    norm2 = sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


# ===== FASTAPI =====
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "running"}

# ===== STATE =====
class AgentState(TypedDict):
    question: str
    plan: str
    tool_output: str
    memory_context: str
    final_answer: str
    timestamps: list   # ✅ ADD THIS

# ===== HELPERS =====
def extract_video_id(url):
    match = re.search(r"(?:v=|youtu.be/)([^&?/]{11})", url)
    return match.group(1) if match else None

def clean_youtube_url(url):
    match = re.search(r"(?:v=|youtu.be/)([^&?/]{11})", url)
    if not match:
        return None
    return f"https://www.youtube.com/watch?v={match.group(1)}"

# ===== GCS TRANSCRIPT STORAGE =====
def save_transcript(video_id, transcript):
    try:
        bucket = get_bucket()
        blob = bucket.blob(f"transcripts/{video_id}.txt")
        blob.upload_from_string(transcript)
        print("✅ Transcript saved")
    except Exception as e:
        print("❌ Save transcript error:", e)

def load_transcript(video_id):
    try:
        bucket = get_bucket()
        blob = bucket.blob(f"transcripts/{video_id}.txt")
        if blob.exists():
            print("📥 Loading transcript from GCS")
            return blob.download_as_text()
        return None
    except Exception as e:
        print("❌ Load transcript error:", e)
        return None

# ===== VECTOR STORE =====
def load_vectorstore(video_id):
    try:
        bucket = get_bucket()
        embeddings = get_embeddings()
        blob = bucket.blob(f"vectors/{video_id}.zip")

        if not blob.exists():
            return None

        zip_path = f"/tmp/{video_id}.zip"
        extract_path = f"/tmp/{video_id}"

        blob.download_to_filename(zip_path)
        shutil.unpack_archive(zip_path, extract_path)

        return FAISS.load_local(
            extract_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print("❌ Load vector error:", e)
        return None

def save_vectorstore(video_id, store):
    try:
        print("📦 Saving vector:", video_id)

        bucket = get_bucket()
        tmp_dir = f"/tmp/{video_id}"

        store.save_local(tmp_dir)
        shutil.make_archive(tmp_dir, "zip", tmp_dir)

        bucket.blob(f"vectors/{video_id}.zip").upload_from_filename(f"{tmp_dir}.zip")

        print("✅ Vector saved to GCS")
    except Exception as e:
        print("❌ GCS vector save error:", e)

# ===== MEMORY =====
def load_memory(user_id):
    try:
        bucket = get_bucket()
        embeddings = get_embeddings()

        blob = bucket.blob(f"memory/{user_id}.zip")
        if not blob.exists():
            return None

        zip_path = f"/tmp/{user_id}.zip"
        extract_path = f"/tmp/{user_id}"

        blob.download_to_filename(zip_path)
        shutil.unpack_archive(zip_path, extract_path)

        return FAISS.load_local(
            extract_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print("❌ Load memory error:", e)
        return None

def save_memory(user_id, question, answer):
    try:
        print("💾 Saving memory:", user_id)

        bucket = get_bucket()
        embeddings = get_embeddings()

        tmp_dir = f"/tmp/{user_id}"
        doc = Document(page_content=f"Q: {question}\nA: {answer}")

        store = load_memory(user_id)
        if store:
            store.add_documents([doc])
        else:
            store = FAISS.from_documents([doc], embeddings)

        store.save_local(tmp_dir)
        shutil.make_archive(tmp_dir, "zip", tmp_dir)

        bucket.blob(f"memory/{user_id}.zip").upload_from_filename(f"{tmp_dir}.zip")

        print("✅ Memory saved to GCS")
    except Exception as e:
        print("❌ GCS memory save error:", e)

def retrieve_memory(user_id, query):
    try:
        store = load_memory(user_id)
        if not store:
            return ""
        docs = store.similarity_search(query, k=3)
        return "\n".join([d.page_content for d in docs])
    except Exception as e:
        print("❌ Retrieve memory error:", e)
        return ""

# ===== CHAT HISTORY =====
def save_chat_history(session_id, question, answer):
    try:
        bucket = get_bucket()
        blob = bucket.blob(f"chat_history/{session_id}.json")

        history = []

        if blob.exists():
            history = json.loads(blob.download_as_text())


        history.append({
            "question": question,
            "answer": answer,
            "timestamp": str(datetime.utcnow())
        })

        blob.upload_from_string(json.dumps(history, indent=2))

        print("💾 Chat history saved")

    except Exception as e:
        print("❌ Chat history error:", e)


def load_chat_history(session_id):
    try:
        bucket = get_bucket()
        blob = bucket.blob(f"chat_history/{session_id}.json")

        if not blob.exists():
            return []

        return json.loads(blob.download_as_text())

    except Exception as e:
        print("❌ Load history error:", e)
        return []



# ===== NORMALIZER =====
def normalize_apify_data(data):
    print("🧠 Normalizing Apify response...")

    try:
        print("🔎 Incoming type:", type(data))

        # Case 1: list
        if isinstance(data, list):
            print("📦 Data is LIST, length:", len(data))

            if len(data) > 0:
                first = data[0]
                print("🔎 First element keys:", list(first.keys()) if isinstance(first, dict) else "Not dict")

                if isinstance(first, dict):

                    # ✅ MOST IMPORTANT CASE (YOUR CASE)
                    if "data" in first:
                        print("✅ Found 'data' inside list[0]")
                        return first["data"]

                    if "items" in first:
                        print("✅ Found 'items' inside list[0]")
                        return first["items"]

                    if "text" in first:
                        print("✅ Already normalized list")
                        return data

        # Case 2: dict
        if isinstance(data, dict):
            print("📦 Data is DICT, keys:", list(data.keys()))

            if "items" in data:
                print("✅ Found 'items' in dict")
                return data["items"]

            if "data" in data:
                if isinstance(data["data"], list):
                    print("✅ Found 'data' list in dict")
                    return data["data"]

                if isinstance(data["data"], dict) and "items" in data["data"]:
                    print("✅ Found nested data->items")
                    return data["data"]["items"]

        print("❌ Could not normalize Apify response")
        return None

    except Exception as e:
        print("❌ Normalize error:", e)
        return None

# ===== PARSER =====
def parse_apify_transcript(data):



    documents = []

    total_items = len(data)
    valid_items = 0
    skipped_items = 0

    print("🔎 Total items received:", total_items)

    # ===== 🔥 CHUNK SCORING FUNCTION =====
    def score_chunk(text):
        words = text.split()

        score = 0
        score += min(len(words) / 20, 2)
        score += len(set(words)) / max(len(words), 1)

        if "." in text:
            score += 1

        if len(words) < 6:
            score -= 1

        return score

    # ===== 🔥 PARSE RAW DATA =====
    for i, item in enumerate(data):
        try:
            if not isinstance(item, dict):
                skipped_items += 1
                continue

            text = (
                item.get("text")
                or item.get("caption")
                or item.get("content")
                or ""
            )

            if not text or not isinstance(text, str):
                skipped_items += 1
                continue

            clean_text = text.strip().replace("\n", " ")

            # 🔥 CLEAN NOISE
            clean_text = re.sub(r"\{.*?\}", "", clean_text)
            clean_text = re.sub(r"\[.*?\]", "", clean_text)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()

            words = clean_text.split()

            if len(words) < 5:
                continue

            if len(set(words)) < 3:
                continue

            alpha_ratio = sum(c.isalpha() for c in clean_text) / max(len(clean_text), 1)
            if alpha_ratio < 0.5:
                continue

            filler_phrases = [
                "hello", "welcome", "thank you", "subscribe",
                "like this video", "namaskar", "good morning"
            ]

            if any(fp in clean_text.lower() for fp in filler_phrases):
                continue

            start = item.get("start") or item.get("startTime")

            try:
                start_val = float(start) if start is not None else None
            except:
                start_val = None

            score = score_chunk(clean_text)

            if score > 1.2:
                documents.append(
                    Document(
                        page_content=clean_text,
                        metadata={
                            "start": start_val,
                            "score": score
                        }
                    )
                )
                valid_items += 1

                if valid_items <= 5:
                    print(f"✅ Valid {valid_items}: [{start_val}] {clean_text[:80]}")

        except Exception as e:
            skipped_items += 1
            print("⚠️ Error parsing item:", e)

    print("📊 PARSE STATS:")
    print("   Total items:", total_items)
    print("   Valid items:", valid_items)
    print("   Skipped items:", skipped_items)
    print("🧠 Documents created:", len(documents))

    # ===== 🔥 SEMANTIC MERGE (FINAL FIX) =====

    merged_docs = []
    buffer = []
    buffer_start = None

    MAX_GAP = 8
    MAX_WORDS = 120

    def is_incomplete(text):
        text = text.strip()
        if not text:
            return True

        if not re.search(r"[.!?]$", text):
            return True

        if len(text.split()) < 6:
            return True

        return False

    def get_last_timestamp(buf):
        if not buf:
            return None
        return buf[-1].metadata.get("start")

    for d in documents:
        text = d.page_content.strip()
        start = d.metadata.get("start")

        if not text:
            continue

        last_ts = get_last_timestamp(buffer)
        should_merge = False

        if is_incomplete(text):
            should_merge = True

        elif last_ts is not None and start is not None:
            if abs(start - last_ts) <= MAX_GAP:
                should_merge = True

        if should_merge:
            if buffer_start is None:
                buffer_start = start

            buffer.append(d)

            total_words = sum(len(x.page_content.split()) for x in buffer)

            if total_words >= MAX_WORDS:
                merged_docs.append(
                    Document(
                        page_content=" ".join(x.page_content for x in buffer),
                        metadata={"start": buffer_start}
                    )
                )
                buffer = []
                buffer_start = None

        else:
            if buffer:
                merged_docs.append(
                    Document(
                        page_content=" ".join(x.page_content for x in buffer),
                        metadata={"start": buffer_start}
                    )
                )
                buffer = []
                buffer_start = None

            merged_docs.append(d)

    if buffer:
        merged_docs.append(
            Document(
                page_content=" ".join(x.page_content for x in buffer),
                metadata={"start": buffer_start}
            )
        )

    documents = merged_docs

    # ===== 🔥 BUILD TRANSCRIPT (FIXED) =====
    transcript_lines = []

    for doc in documents:
        start = doc.metadata.get("start")
        text = doc.page_content.strip()

        if not text:
            continue

        if start is not None:
            # 🔥 REMOVE existing timestamps from text first
            clean_text = re.sub(r"\[\d+s\]", "", text).strip()

            transcript_lines.append(f"[{int(start)}s] {clean_text}")
        else:
            clean_text = re.sub(r"\[\d+s\]", "", text).strip()
            transcript_lines.append(clean_text)

    transcript = "\n".join(transcript_lines)

    # ===== VALIDATION =====
    if not documents:
        print("❌ No documents at all → fallback")
        return None, None

    if not transcript or len(transcript.strip()) < 50:
        print("⚠️ Transcript too small → fallback")
        return None, None

    print("🧠 Transcript length:", len(transcript))
    print("🔍 Preview (1000 chars):", transcript[:1000])

    return transcript, documents

# ===== APIFY FETCH =====
def fetch_transcript_apify(video_url):
    try:
        print("🚀 Using Apify for transcript")

        video_url = clean_youtube_url(video_url)

        url = f"https://api.apify.com/v2/acts/pintostudio~youtube-transcript-scraper/run-sync-get-dataset-items?token={APIFY_TOKEN}"
        payload = {"videoUrl": video_url}

        # 🔥 API CALL
        res = requests.post(url, json=payload, timeout=60)

        # 🔍 BASIC LOGS
        print("🔎 Status Code:", res.status_code)
        print("🔎 Raw response (first 500 chars):", res.text[:500])

        # ✅ HANDLE HTTP ERROR PROPERLY
        if not res.ok:
            print("❌ Apify HTTP error:", res.status_code)
            return None, None

        # 🔥 SAFE JSON PARSING (CRITICAL FIX)
        try:
            raw_data = res.json()
            print("✅ JSON parsed successfully")
        except Exception as e:
            print("❌ JSON parse error:", e)
            print("🔎 Raw text (500 chars):", res.text[:500])
            return None, None

        print("🧠 Raw JSON type:", type(raw_data))

        # 🔥 NORMALIZE DATA (VERY IMPORTANT)
        data = normalize_apify_data(raw_data)

        if not data:
            print("❌ Failed to extract transcript list after normalization")
            return None, None

        print("✅ Normalized items count:", len(data))

        # 🔍 SHOW SAMPLE
        if len(data) > 0:
            print("🔎 First normalized item:", data[0])
            print("🔎 Last normalized item:", data[-1])

        # 🔥 PARSE TRANSCRIPT
        transcript, documents = parse_apify_transcript(data)

        # ❌ HANDLE PARSE FAILURE
        if not transcript:
            print("⚠️ Parser returned empty → fallback will trigger")
            return None, None

        print("✅ Transcript successfully parsed")
        print("📊 Final document count:", len(documents))

        return transcript, documents

    except Exception as e:
        print("❌ Apify error:", e)
        return None, None


# ===== VIDEO PROCESS (UPDATED FIXES HERE) =====

# 🔥 GLOBAL CACHE (ADD THIS AT TOP OF FILE ONCE)
VECTOR_CACHE = {}

# ===== VIDEO PROCESS (UPDATED) =====
def get_or_create_vectorstore(video_url):
    vid = extract_video_id(video_url)

    # ===== 🔥 STEP 1: IN-MEMORY CACHE =====
    if vid in VECTOR_CACHE:
        print("⚡ Using in-memory vector cache")
        return VECTOR_CACHE[vid]

    # ===== STEP 2: LOAD FROM GCS =====
    store = load_vectorstore(vid)
    if store:
        print("📦 Loaded vector from GCS")

        VECTOR_CACHE[vid] = store  # 🔥 cache it
        return store

    # ===== STEP 3: LOAD TRANSCRIPT =====
    transcript = load_transcript(vid)

    if transcript:
        print("⚡ Using cached transcript")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        docs = splitter.create_documents([transcript])

        for d in docs:
            matches = re.findall(r"\[(\d+)s\]", d.page_content)
            if matches:
                d.metadata["start"] = int(matches[0])  # still first, but safer
            else:
                d.metadata.pop("start", None)
    else:
        transcript, docs = fetch_transcript_apify(video_url)

        if not transcript:
            print("⚠️ Using fallback (NOT saving)")
            transcript = wikipedia.summary("YouTube video", sentences=3)
            docs = [Document(page_content=transcript, metadata={})]
        else:
            save_transcript(vid, transcript)

    # ===== STEP 4: EMBEDDINGS =====
    embeddings = get_embeddings()

    # 🔥 IMPROVED CHUNKING
    split_docs = RecursiveCharacterTextSplitter(
        chunk_size=800,        # 🔥 improved
        chunk_overlap=100
    ).split_documents(docs)

    print("📊 Total chunks created:", len(split_docs))

    # ===== STEP 5: CREATE VECTOR STORE =====
    store = FAISS.from_documents(split_docs, embeddings)

    # ===== STEP 6: SAVE + CACHE =====
    save_vectorstore(vid, store)

    VECTOR_CACHE[vid] = store  # 🔥 cache in memory

    return store


def keyword_search(transcript, query):
    words = query.lower().split()
    lines = transcript.split("\n")

    results = []

    for line in lines:
        score = sum(1 for w in words if w in line.lower())

        # 🔥 Match if at least half words match
        if score >= max(1, len(words) // 2):
            results.append(line)

    return results[:5]

# ===== AGENT =====
def build_graph(store, user_id, transcript):
    graph = StateGraph(AgentState)

    # ===== PLANNER =====
    @traceable(name="planner")
    def planner(state):
        question = state["question"]

        prompt = f"""
        Decide how to answer this question.

        Question: {question}

        Options:
        - retrieve → use transcript to answer specific questions
        - summarize → give overall summary of the video
        - memory → use past conversation

        RULES:

        - If the question asks about specific facts, names, numbers, or details → choose "retrieve"
        - If the question asks for explanation, overview, meaning, summary, or general understanding → choose "summarize"
        - If the question depends on previous conversation → choose "memory"
        
        - If the question requires understanding the overall context or multiple parts of the video → choose "summarize"
        - If the question can be answered using a few specific lines → choose "retrieve"
        
        IMPORTANT:
        - Prefer "summarize" when the intent is to understand the full video or big picture
        - Prefer "retrieve" when the intent is to extract precise information

        Return ONLY one word.
        """

        decision = safe_llm_call(prompt).strip().lower()

        # ===== 🔥 NEW: QUESTION CLASSIFICATION (ADD HERE) =====

        classification_prompt = f"""
        Classify this question:

        Question: {question}

        Options:
        - factual
        - explanatory
        - conversational

        Return ONLY one word.
        """

        q_type = safe_llm_call(classification_prompt).strip().lower()

        print("🧠 Question Type:", q_type)

        # ===== 🔥 MAP TYPE TO PLAN =====
        if q_type == "factual":
            decision = "retrieve"
        elif q_type == "explanatory":
            decision = "summarize"
        elif q_type == "conversational":
            decision = "memory"

        # ===== VALIDATION =====
        valid_plans = ["retrieve", "summarize", "memory"]

        if decision not in valid_plans:
            print("⚠️ Invalid planner output → defaulting to retrieve")
            decision = "retrieve"



        # ===== 🔥 IMPROVED LENGTH-BASED CORRECTION =====
        q_len = len(question.split())

        # only override for VERY short + weak queries
        if q_len <= 4 and decision == "summarize":
            print("⚠️ Very short query → forcing retrieve")
            decision = "retrieve"

        print("🧠 Plan:", decision)

        state["plan"] = decision
        return state

    # ===== 🔥 CONTEXT COMPRESSION =====
    def compress_context(context, max_lines=20):
        lines = context.split("\n")

        # ✅ keep only non-empty lines
        lines = [l for l in lines if l.strip()]

        # ✅ keep original order
        if len(lines) > max_lines:
            half = max_lines // 2
            lines = lines[:half] + lines[-half:]
        return "\n".join(lines)

    # ===== TOOL =====
    @traceable(name="tool")
    def tool(state):
        plan = state.get("plan", "retrieve")
        query = state["question"]

        # ===== 🔥 EXTRACT REAL TIMESTAMP FROM CHUNK =====
        def extract_timestamp_from_chunk(text):
            matches = re.findall(r"\[(\d+)s\]", text)
            if matches:
                return int(matches[0])  # take first timestamp
            return None

        if plan == "summarize":
            print("📘 Using FULL transcript (smart coverage + importance)")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )

            chunks = splitter.split_text(transcript)

            total_chunks = len(chunks)
            max_chunks = min(total_chunks, 10)

            selected_chunks = []

            # ===== 🔥 STEP 1: COVERAGE (start → middle → end) =====
            step = max(1, total_chunks // max_chunks)

            for i in range(0, total_chunks, step):
                selected_chunks.append(chunks[i])
                if len(selected_chunks) >= max_chunks // 2:
                    break

            # ===== 🔥 STEP 2: IMPORTANCE (longer = richer info) =====
            remaining = sorted(chunks, key=lambda x: len(x), reverse=True)

            for c in remaining:
                if c not in selected_chunks:
                    selected_chunks.append(c)
                if len(selected_chunks) >= max_chunks:
                    break

            # ===== 🔥 FINAL DOCS =====
            # ===== 🔥 FINAL DOCS WITH REAL TIMESTAMPS =====
            docs = [
                Document(
                    page_content=chunk,
                    metadata={"start": extract_timestamp_from_chunk(chunk)}
                )
                for chunk in selected_chunks
            ]
            state["retrieval_count"] = len(docs)

        elif plan == "memory":
            print("🧠 Using MEMORY")
            docs = [
                Document(
                    page_content=state.get("memory_context", "")
                )
            ]

        else:
            print("🔎 Using VECTOR SEARCH")

            # ===== 🔥 ADD HERE =====
            total_chunks = store.index.ntotal if hasattr(store, "index") else 50

            print("📊 Total vector chunks:", total_chunks)

            def get_base_k(total_chunks):
                if total_chunks <= 20:
                    return 4  # small video
                elif total_chunks <= 50:
                    return 6
                elif total_chunks <= 150:
                    return 10
                else:
                    return 15  # large video

            base_k = get_base_k(total_chunks)

            query_lower = query.lower()
            query_words = query_lower.split()
            query_len = len(query_words)

            # ===== 🔥 GENERIC QUERY ANALYSIS (NO HARDCODING) =====

            # short query → usually vague
            is_short = query_len <= 5

            # low information query (repeated / weak words)
            is_low_info = len(set(query_words)) <= 3

            # question structure detection (generic)
            question_starters = {"who", "what", "which", "whom", "whose", "where", "when"}
            starts_like_question = query_words[0] in question_starters if query_words else False


            # ===== 🔥 FINAL K (FIXED) =====
            k = base_k

            # slight boost for very short queries
            if len(query.split()) <= 3:
                k += 2

            k = min(k, total_chunks)

            # ===== 🔥 ENTITY-LIKE DETECTION (GENERIC) =====
            is_entity_like = is_short and starts_like_question

            search_query = query

            if is_entity_like:
                print("🧠 Entity-like query → boosting semantic meaning")
                search_query = query + " identity role context details"

            # ===== 🔥 FINAL SEMANTIC SEARCH =====
            semantic_docs = store.similarity_search(search_query, k=k)


            # ===== 🔥 LIMIT KEYWORD IMPACT =====
            keyword_results = keyword_search(transcript, query)

            keyword_docs = []

            for txt in keyword_results[:2]:  # 🔥 LIMIT TO 2 ONLY
                match = re.search(r"\[(\d+)s\]", txt)
                start = int(match.group(1)) if match else None

                keyword_docs.append(
                    Document(page_content=txt, metadata={"start": start})
                )

            # ===== 🔥 STEP 1: COMBINE (SEMANTIC PRIORITY) =====
            docs = semantic_docs + keyword_docs

            # ===== 🔥 STEP 2: REMOVE DUPLICATES =====
            seen = set()
            unique_docs = []

            for d in docs:
                content = d.page_content.strip()
                if content not in seen:
                    seen.add(content)
                    unique_docs.append(d)

            # ===== 🔥 STEP 3: SEMANTIC RE-RANK (IMPORTANT FIX) =====
            def semantic_score(doc, query):
                doc_words = set(doc.page_content.lower().split())
                query_words = set(query.lower().split())

                overlap = len(doc_words.intersection(query_words))

                # 🔥 BOOST LONGER + RICHER CONTEXT
                length_bonus = min(len(doc.page_content) / 200, 2)

                return overlap + length_bonus

            # ===== 🔥 STEP 3: SEMANTIC RE-RANK =====
            ranked_docs = sorted(
                unique_docs,
                key=lambda d: semantic_score(d, query),
                reverse=True
            )

            docs = ranked_docs[:k]

            state["retrieval_count"] = len(docs)

            print(f"📊 Final docs after adaptive retrieval: {len(docs)}")

        # 🔥 BOOST MEMORY INTO CONTEXT (ONLY IF RELEVANT)
        if state.get("memory_context"):
            context = "Previous conversation:\n" + state["memory_context"] + "\n\n"
        else:
            context = ""

        timestamps = []
        timestamps_set = set()  # 🔥 NEW (fast lookup)

        for d in docs:
            ts = d.metadata.get("start")
            content = d.page_content.strip()

            if not content:
                continue

            if ts is not None:
                context += f"[{int(ts)}s] {content}\n"

                # 🔥 OPTIMIZED DUPLICATE CHECK
                if ts not in timestamps_set:
                    timestamps.append(ts)
                    timestamps_set.add(ts)

            else:
                context += f"{content}\n"


        # ✅ HANDLE EMPTY CONTEXT FIRST
        if not context.strip():
            print("⚠️ No relevant context found")
            context = "No relevant transcript found."

        # ===== 🔥 SAVE RAW CONTEXT BEFORE COMPRESSION =====
        raw_context = context

        # ===== 🔥 APPLY CONTEXT COMPRESSION =====
        context = compress_context(context)


        if not context.strip():
            print("❌ EMPTY CONTEXT BUG")

        # 🔥 FORCE SAFE CONTEXT (VERY IMPORTANT)
        if len(context.strip()) < 20:
            context = raw_context
        state["tool_output"] = context

        # ===== 🔥 SET CONTEXT LENGTH =====
        state["context_length"] = len(context)

        # ===== 🔥 ENSURE retrieval_count EXISTS =====
        if "retrieval_count" not in state:
            state["retrieval_count"] = len(docs)

        # ===== 🔥 EXTRACT TIMESTAMPS FROM RAW CONTEXT (NOT COMPRESSED) =====
        valid_timestamps = []

        for line in raw_context.split("\n"):
            match = re.search(r"\[(\d+)s\]", line)
            if match:
                ts = int(match.group(1))
                if ts not in valid_timestamps:  # optional dedup
                    valid_timestamps.append(ts)

        # ===== FINAL OUTPUT =====
        state["tool_output"] = context
        state["timestamps"] = valid_timestamps

        return state


    # ===== REASONING =====
    @traceable(name="reasoning")
    def reasoning_node(state):
        prompt = f"""
        Analyze the transcript carefully and reconstruct complete meaning.

        Context:
        {state["tool_output"]}

        Question:
        {state["question"]}

        STRICT RULES:
        - DO NOT return broken or incomplete sentences
        - Combine related lines into FULL meaningful points
        - If a sentence looks cut → complete it using nearby context
        - Each point must explain a full idea (not fragments)
        - Avoid repeating similar lines
        - Do NOT assume anything outside transcript
        - If information is insufficient → say "Not enough information"

        OUTPUT FORMAT:
        - Bullet points
        - Each bullet must be a complete sentence
        - Each bullet must be understandable on its own
        """

        reasoning = safe_llm_call(prompt)

        # ===== 🔥 REMOVE VERY SHORT / BROKEN LINES =====
        lines = reasoning.split("\n")
        cleaned = []

        for l in lines:
            if len(l.strip()) > 20:
                cleaned.append(l)

        reasoning = "\n".join(cleaned)

        # ===== 🔥 LLM-BASED HALLUCINATION CHECK =====
        def is_reasoning_hallucinated(context, reasoning):
            prompt = f"""
            Check whether this reasoning is fully grounded in the transcript.

            Transcript:
            {context}

            Reasoning:
            {reasoning}

            Return YES if:
            - reasoning includes assumptions not supported by transcript
            - reasoning adds external knowledge
            - reasoning is speculative

            Return NO if:
            - reasoning is fully based on transcript
            - reasoning only combines transcript information

            IMPORTANT:
            - Be strict
            - Even small unsupported assumptions → YES

            Return ONLY YES or NO.
            """

            decision = safe_llm_call(prompt)
            decision_clean = decision.strip().lower()

            print("🧠 Reasoning grounding check:", decision_clean)

            return "yes" in decision_clean

        # ===== 🔥 APPLY CHECK =====
        if reasoning.strip():
            if is_reasoning_hallucinated(state["tool_output"], reasoning):
                print("⚠️ Hallucinated reasoning detected → resetting")
                reasoning = "Reasoning based strictly on available transcript."

        print("🧠 Reasoning:", reasoning[:200])

        state["reasoning"] = reasoning
        return state

    # ===== MEMORY =====
    @traceable(name="memory")
    def memory_node(state):

        memory = retrieve_memory(user_id, state["question"])

        # 🔥 FILTER RELEVANT MEMORY ONLY
        if memory and len(memory.strip()) > 20:
            state["memory_context"] = memory
        else:
            state["memory_context"] = ""

        return state

    # ===== ANSWER =====
    @traceable(name="answer")
    def answer_node(state):
        plan = state.get("plan", "")
        state["needs_fallback"] = False

        # ===== 🔥 DIFFERENT PROMPT FOR SUMMARIZE =====
        if plan == "summarize":

            prompt = f"""
        You are an AI assistant.

        Your task is to EXPLAIN the video clearly using the transcript.

        Transcript:
        {state['tool_output']}

        Reasoning:
        {state.get('reasoning', '')}

        IMPORTANT INSTRUCTIONS:

        - Combine fragmented lines into meaningful explanations
        - Explain the idea, not just repeat sentences
        - Connect related points logically
        - DO NOT just copy transcript lines
        
        - Keep answer concise and structured
        - Ensure the answer gives a COMPLETE understanding of the video
        - Include all key events, claims, and context discussed

        STRICT RULES:

        - ONLY use transcript (no outside knowledge)
        - If something is missing → skip it
        - Do NOT hallucinate
        
        OUTPUT FORMAT (VERY STRICT):
        You MUST follow ALL rules:
        
        1. Output ONLY bullet points
        2. Maximum 6 bullets (not more)
        3. Each bullet = ONE idea ONLY
        4. Each bullet = max 2 lines
        5. Do NOT write paragraphs
        6. Do NOT merge multiple ideas
        7. Do NOT repeat similar points
        8. Concise structured answers ONLY

        Question:
        {state['question']}
        """



        # ===== 🔥 NORMAL QA PROMPT =====
        else:
            prompt = f"""
        You are an AI assistant.
    
        You MUST answer ONLY using the provided transcript.
    
        Transcript:
        {state['tool_output']}
    
        Memory:
        {state['memory_context']}
    
        Reasoning:
        {state.get('reasoning', '')}
    
        Question:
        {state['question']}
    
        STRICT RULES (VERY IMPORTANT):
    
        1. DO NOT use any outside knowledge
        2. DO NOT assume or generalize or hallucinate
        3. Use ONLY the provided transcript as source of truth
        4. You MAY combine multiple transcript lines to infer a complete answer
        5. Combine lines ONLY if they clearly refer to the same subject
        6. If the answer is truly absent → say: "Not mentioned in the video"
    
        7. Every point MUST be supported by transcript text
        8. Timestamp rules (STRICT):
        - Format ONLY like: [mm:ss]
        - DO NOT write: "at [mm:ss]" or "as mentioned at"
        - DO NOT add any words before timestamps
        - Timestamp must be at end of bullet point
    
        9. DO NOT invent structure like:
           - "Lesson 1, Lesson 2"
           - unless explicitly present in transcript
    
        10. DO NOT give fake timestamps like 0:00, 5:00, etc.
        11. Use memory ONLY if it is directly relevant to the question
        12. If question refers to past conversation → prioritize memory over transcript
        13. You MUST ONLY use timestamps that appear in the transcript context
        14. If you are not 100% sure → DO NOT include timestamp
        15. NEVER generate timestamps on your own
        
        OUTPUT FORMAT (VERY STRICT):

        You MUST follow ALL rules:
        
        1. Output ONLY bullet points
        2. Maximum 4 bullets (not more)
        3. Each bullet = ONE idea ONLY
        4. Each bullet = max 2 lines
        5. Do NOT write paragraphs
        6. Do NOT merge multiple ideas
        7. Do NOT repeat similar points
        9. DO NOT generate timestamps.
        10. Timestamps will be added automatically.
        11. Use real transcript wording where possible
        12. Concise structured answers ONLY
        """

        # ===== GENERATE ANSWER =====
        # 🔥 SELECT TASK BASED ON PLAN
        task_type = "summary" if plan == "summarize" else "qa"

        answer = safe_llm_call(prompt, task=task_type)

        if "error generating response" in answer.lower():
            answer = "Not mentioned in the video"

        # 🔥 EXPAND TOO SHORT SUMMARIES
        if plan == "summarize" and len(answer.strip()) < 80:
            print("⚠️ Expanding short summary")

            expand_prompt = f"""
            Expand this explanation to make it more complete and meaningful.

            Answer:
            {answer}

            Rules:
            - Add missing context from transcript
            - Keep it clear and structured
            - Do NOT add outside knowledge
            """

            answer = safe_llm_call(expand_prompt, task="summary")


        # ===== 🔥 FORCE TIMESTAMP ALIGNMENT (FOR BOTH summarize + retrieve) =====

        context_lines = state["tool_output"].split("\n")
        # ===== 🔥 CLEAN + NORMALIZE BULLETS (IMPORTANT FIX) =====

        raw_lines = answer.split("\n")

        answer_lines = []

        for line in raw_lines:
            line = line.strip()

            if not line:
                continue

            # remove bullet symbols
            line = re.sub(r"^[•\-]\s*", "", line)

            # 🔥 split if multiple ideas joined by common separators
            parts = re.split(r"\s+\|\s+|\s+-\s+|\.\s+(?=[A-Z])", line)

            for p in parts:
                p = p.strip()

                # ignore very small fragments
                if len(p.split()) < 4:
                    continue

                answer_lines.append(p)

        embeddings = get_embeddings()

        # ===== 🔥 BATCH ANSWER EMBEDDINGS =====
        answer_texts = [ans.lower()[:300] for ans in answer_lines]

        try:
            answer_embeddings = embeddings.embed_documents(answer_texts)
        except Exception as e:
            print("❌ Answer embedding error:", e)
            answer_embeddings = [None] * len(answer_lines)


        # ===== 🔥 BATCHED CONTEXT EMBEDDINGS (OPTIMIZED) =====

        context_embeddings = []
        context_timestamps = []
        filtered_texts = []

        # Step 1: collect valid lines + timestamps
        for line in context_lines:
            match = re.search(r"\[(\d+)s\]", line)
            if not match:
                continue

            ts = match.group(1)
            text = line.lower().strip()

            if not text:
                continue

            filtered_texts.append(text[:500])  # keep aligned order
            context_timestamps.append(ts)

        # Step 2: batch embedding (single API call)
        try:
            if filtered_texts:
                context_embeddings = embeddings.embed_documents(filtered_texts)
            else:
                context_embeddings = []
        except Exception as e:
            print("❌ Context embedding error:", e)
            context_embeddings = []

        new_answer = []

        # ===== 🔥 PREVENT DUPLICATES =====
        used_timestamps = set()

        # ===== 🔥 UPDATED LOOP WITH BATCH =====
        for idx, ans in enumerate(answer_lines):

            matched_ts = None

            ans_emb = answer_embeddings[idx]

            if ans_emb and context_embeddings:

                # ===== 🔥 BEST MATCH ONLY (IMPORTANT FIX) =====
                best_score = 0
                best_idx = None

                for i, ctx_emb in enumerate(context_embeddings):
                    score = cosine_similarity(ans_emb, ctx_emb)

                    if score > best_score:
                        best_score = score
                        best_idx = i

                # 🔥 APPLY THRESHOLD
                if best_score >= 0.65 and best_idx is not None:

                    candidate_ts = context_timestamps[best_idx]

                    if candidate_ts != "0" and candidate_ts not in used_timestamps:
                        matched_ts = candidate_ts
                        used_timestamps.add(candidate_ts)

            # ===== 🔥 CLEAN EXISTING TIMESTAMPS =====
            ans_clean = re.sub(r"\[\d+:\d{2}\]", "", ans)  # remove [mm:ss]
            ans_clean = re.sub(r"\[\d+s\]", "", ans_clean)  # remove [123s]
            ans_clean = ans_clean.strip()

            ans_clean = re.sub(r"\s+", " ", ans_clean)

            # ===== 🔥 ADD FINAL TIMESTAMP =====
            if matched_ts:
                new_answer.append(f"{ans_clean} [{matched_ts}s]")
            else:
                new_answer.append(ans_clean)

        answer = "\n".join(new_answer)


        # ===== 🔥 FINAL TIMESTAMP FIX BLOCK (COMPLETE) =====

        # ---------- 1. sec → mm:ss ----------
        def sec_to_mmss(sec):
            sec = int(sec)
            m = sec // 60
            s = sec % 60
            return f"{m}:{s:02d}"

        # ---------- 2. convert [123s] → [mm:ss] ----------
        def convert_timestamps(text):
            matches = re.findall(r"\[(\d+)s\]", text)

            for m in matches:
                mmss = sec_to_mmss(int(m))
                text = text.replace(f"[{m}s]", f"[{mmss}]")

            return text

        # ---------- 3. clean formatting ----------
        def clean_timestamp_format(text):
            text = re.sub(r"\bat\s*\[(\d+:\d{2})\]", r"[\1]", text)
            text = re.sub(r"\bas noted at\s*\[(\d+:\d{2})\]", r"[\1]", text)
            text = re.sub(r"\bat time\s*\[(\d+:\d{2})\]", r"[\1]", text)
            text = re.sub(r"\[(\d+:\d{2})\]\.", r"[\1]", text)
            return text

        # ---------- 4. REMOVE FAKE TIMESTAMPS (🔥 MOST IMPORTANT) ----------
        def mmss_to_sec(m, s):
            return int(m) * 60 + int(s)

        def remove_fake_timestamps(answer, real_timestamps):
            matches = re.findall(r"\[(\d+):(\d{2})\]", answer)

            for m, s in matches:
                sec = mmss_to_sec(m, s)

                # ✅ allow small tolerance (±5 sec)
                valid = any(abs(sec - int(t)) <= 5 for t in real_timestamps)

                if not valid:
                    print(f"❌ Removing fake timestamp: [{m}:{s}]")
                    answer = answer.replace(f"[{m}:{s}]", "")

            return answer

        # ---------- APPLY ALL STEPS ----------
        answer = convert_timestamps(answer)
        answer = clean_timestamp_format(answer)

        # 🔥 use REAL timestamps from tool()
        real_ts = state.get("timestamps", [])

        answer = remove_fake_timestamps(answer, real_ts)
        # ===== END FIX =====


        # ===== NORMALIZE "NOT FOUND" =====
        low = answer.lower()

        def is_answer_sufficient(question, context, answer):
            prompt = f"""
            Evaluate if the answer correctly and sufficiently answers the question.

            Question:
            {question}

            Transcript:
            {context}

            Answer:
            {answer}

            Return:
            SUFFICIENT or INSUFFICIENT

            Rules:
            - SUFFICIENT → answer clearly addresses the question
            - INSUFFICIENT → answer is vague, incomplete, or irrelevant
            - Even partial but meaningful → SUFFICIENT
            """

            decision = safe_llm_call(prompt).strip().lower()

            print("🧠 Answer quality decision:", decision)

            return "sufficient" in decision

        if plan != "summarize":

            sufficient = is_answer_sufficient(
                state["question"],
                state["tool_output"],
                answer
            )

            if not sufficient:
                print("⚠️ Answer insufficient → keeping for fallback")
                state["needs_fallback"] = True

        # ===== REMOVE HALLUCINATED STRUCTURE =====
        low = answer.lower()

        def has_hallucinated_structure(answer):
            prompt = f"""
            Check if this answer contains structure NOT present in transcript.

            Answer:
            {answer}

            Return YES if:
            - fake sections like "Lesson 1", "Conclusion" are invented

            Return NO if:
            - structure is natural or harmless
            """

            decision = safe_llm_call(prompt).strip().lower()
            return "yes" in decision

        if has_hallucinated_structure(answer):
            print("⚠️ Hallucinated structure detected")
            state["needs_fallback"] = True

        # ===== 🔥 REAL CONFIDENCE CALCULATION =====

        def compute_confidence(context, answer):
            if not context or not answer:
                return 0.0

            context_words = set(context.lower().split())
            answer_words = set(answer.lower().split())

            # 🔹 overlap score
            overlap = len(context_words.intersection(answer_words))
            total = max(len(answer_words), 1)
            overlap_score = overlap / total

            # 🔹 length score (avoid tiny answers)
            length_score = min(len(answer) / 500, 1)

            # 🔹 penalty for "not mentioned"
            penalty = 0
            if "not mentioned" in answer.lower():
                penalty = 0.5

            # 🔹 final score
            confidence = (0.7 * overlap_score) + (0.3 * length_score)
            confidence = confidence - penalty

            return max(0.0, min(confidence, 1.0))

        # ===== APPLY =====
        state["confidence"] = compute_confidence(
            state.get("tool_output", ""),
            answer
        )

        # ===== 🔥 CONTEXT QUALITY SCORE (EMBEDDING BASED) =====

        def compute_context_quality(context, answer):
            try:
                if not context or not answer:
                    return 0.0

                embeddings = get_embeddings()

                embs = embeddings.embed_documents([
                    context[:2000],
                    answer[:1000]
                ])

                context_emb = embs[0]
                answer_emb = embs[1]

                score = cosine_similarity(context_emb, answer_emb)

                return round(score, 3)

            except Exception as e:
                print("❌ Context quality error:", e)
                return 0.0

        # ===== APPLY =====
        state["context_quality"] = compute_context_quality(
            state.get("tool_output", ""),
            answer
        )
        # ===== 🔥 FINAL RELIABILITY SCORE =====
        final_score = (
                0.5 * state["confidence"] +
                0.5 * state["context_quality"]
        )

        state["final_score"] = round(final_score, 3)


        # ===== ENSURE CONTEXT LENGTH =====
        state["context_length"] = state.get("context_length", 0)

        state["final_answer"] = answer
        state["answer_length"] = len(answer)

        state["metrics"] = {
            "confidence": state["confidence"],
            "context_length": state["context_length"],
            "context_quality": state["context_quality"],
            "final_score": state["final_score"],
            "answer_length": state["answer_length"],
            "retrieval_count": state.get("retrieval_count", 0)
        }

        print("🔥 ANSWER NODE context_length:", state.get("context_length"))
        print("🔥 ANSWER NODE confidence:", state.get("confidence"))

        state["debug"] = {
            "plan": state.get("plan"),
            "needs_fallback": state.get("needs_fallback"),
        }
        state["trace"] = {
            "plan": state.get("plan"),
            "retrieval_count": state.get("retrieval_count"),
            "fallback": state.get("needs_fallback")
        }

        return state




    # ===== WIKIPEDIA FALLBACK =====
    @traceable(name="fallback")
    def fallback_node(state):

        plan = state.get("plan", "")

        # 🚫 DO NOT FALLBACK FOR SUMMARIZE
        if plan == "summarize":
            return state

        question = state["question"]
        answer = state["final_answer"]
        context = state.get("tool_output", "")

        # ===== 🔥 CONFIDENCE FUNCTION (KEEP) =====
        def get_confidence(question, context, answer):
            prompt = f"""
            Rate how well the answer is supported by the transcript.

            Question:
            {question}

            Transcript:
            {context}

            Answer:
            {answer}

            Rules:
            - 1 = fully supported AND complete answer
            - 0.5 = partially supported OR incomplete
            - 0 = not supported at all

            IMPORTANT:
            - If answer is cut, vague, or incomplete → DO NOT give 1

            Return ONLY a number.
            """
            score = safe_llm_call(prompt)

            try:
                return float(score.strip())
            except:
                return 0.0

        # ===== COMPUTE CONFIDENCE =====
        confidence = get_confidence(question, context, answer)

        # 🔥 FIX: do NOT override existing confidence
        state["confidence"] = min(
            state.get("confidence", 1.0),
            confidence
        )
        state["metrics"] = {
            "confidence": state["confidence"],
            "context_length": state.get("context_length", 0),
            "context_quality": state.get("context_quality", 0.0),
            "final_score": state.get("final_score", 0.0),
            "answer_length": len(state.get("final_answer", "")),
            "retrieval_count": state.get("retrieval_count", 0)
        }

        print("📊 Confidence:", confidence)

        # ===== 🚫 SKIP IF GOOD ANSWER =====
        # Confidence should NOT block fallback alone
        if confidence >= 0.7:
            print("⚠️ High confidence — but checking usefulness")

        # ===== 🔥 SEMANTIC FALLBACK DECISION (NEW) =====
        def should_fallback(question, context, answer):
            prompt = f"""
            Decide if we should use external knowledge (Wikipedia/Web) 
            to improve the answer.

            Question:
            {question}

            Transcript:
            {context}

            Answer:
            {answer}

            Evaluate based on USER USEFULNESS (not correctness).

            Return YES if:
            - Answer does not actually answer the user's question
            - Answer lacks key information needed by the user
            - Answer is incomplete or unhelpful
            - Answer only says information is missing

            Return NO if:
            - Answer fully satisfies the user's intent
            - Answer provides meaningful information

            IMPORTANT:
            Even if the answer is factually correct,
            if it is NOT useful → return YES

            Return ONLY YES or NO.
            """

            decision = safe_llm_call(prompt)

            decision_clean = decision.strip().lower()

            print("🧠 Raw fallback decision:", decision)
            print("🧠 Clean decision:", decision_clean)
            if not decision_clean:
                print("⚠️ Empty fallback decision → forcing fallback")
                return True

            # ✅ robust check (NOT hardcoded — handles LLM variability)
            return decision_clean.startswith("yes")

        semantic_fallback = should_fallback(question, context, answer)

        if not semantic_fallback:
            print("⛔ Fallback not triggered")
            return state

        print("🌍 Wikipedia fallback triggered")

        try:
            query = question

            # ===== 🔍 SEARCH =====
            search_results = wikipedia.search(query)

            if not search_results:
                print("⚠️ No Wikipedia results found")
                return state

            best_match = search_results[0]
            print("🔎 Wikipedia best match:", best_match)

            # ===== 🔥 SEMANTIC ANSWER STYLE (NEW) =====
            def get_answer_style(question):
                prompt = f"""
                Decide the answer style.

                Question:
                {question}

                Rules:
                - SHORT → 1-2 lines (facts)
                - DETAILED → explanation

                Return ONLY SHORT or DETAILED.
                """
                style = safe_llm_call(prompt).strip().lower()
                return style

            style = get_answer_style(query)

            # ===== 🔥 FETCH WIKI (UPDATED) =====
            try:
                if style == "short":
                    wiki = wikipedia.summary(best_match, sentences=2)
                else:
                    wiki = wikipedia.summary(best_match, sentences=6)

            except wikipedia.exceptions.DisambiguationError as e:
                option = e.options[0]
                print("🔁 Using option:", option)

                if style == "short":
                    wiki = wikipedia.summary(option, sentences=2)
                else:
                    wiki = wikipedia.summary(option, sentences=6)

            except wikipedia.exceptions.PageError:
                print("❌ Page not found")
                return state

            # ===== 🔥 FORMAT ANSWER =====
            prompt = f"""
            Answer the question using this Wikipedia content.

            Content:
            {wiki}

            Question:
            {query}

            Rules:
            - Be clear and direct
            - Do NOT add unnecessary details
            - If factual → keep short
            - If explanatory → explain properly
            """

            new_answer = safe_llm_call(prompt, task="summary")

            # ===== 🔥 SMART MERGE / REPLACE =====

            needs_replace = (confidence < 0.3)

            if needs_replace:
                # 🔥 FULL REPLACEMENT
                state["final_answer"] = (
                        new_answer + "\n\n📚 Source: Wikipedia"
                )
                state["needs_fallback"] = False
            else:
                # 🔥 APPEND (answer already useful)
                state["final_answer"] = (
                        answer + "\n\n---\n\n"
                                 "Additional context from Wikipedia:\n"
                        + new_answer
                        + "\n\n📚 Source: Wikipedia"
                )
                state["needs_fallback"] = False

        except Exception as e:
            print("❌ Wikipedia fallback error:", e)

        # ==============================
        # 🔥 NEW: TAVILY FALLBACK
        # ==============================

        # ==============================
        # 🔥 NEW: TAVILY FALLBACK (FIXED)
        # ==============================
        # ✅ ADD THIS BLOCK HERE
        if not state.get("needs_fallback"):
            print("⛔ Skipping Tavily (fallback already resolved)")
            return state
        try:
            print("🌐 Checking Tavily fallback...")

            query = state["question"]
            current_answer = state["final_answer"]

            # 🔥 USE SEMANTIC DECISION (NO HARDCODING)
            tavily_needed = should_fallback(
                query,
                context,
                current_answer
            )

            if tavily_needed:

                print("🌐 Tavily fallback triggered")

                web_data = search_tavily(query)

                if web_data:
                    prompt = f"""
                    Answer using this web data.

                    Data:
                    {web_data}

                    Question:
                    {query}

                    Rules:
                    - Be accurate
                    - Keep it concise
                    - Do NOT hallucinate
                    """

                    web_answer = safe_llm_call(prompt, task="summary")

                    state["final_answer"] += (
                            "\n\n---\n\n🌐 Additional info (Web):\n"
                            + web_answer
                    )

                    print("✅ Tavily answer used")

        except Exception as e:
            print("❌ Tavily fallback error:", e)

        return state

    # ===== CRITIC =====
    @traceable(name="critic")
    def critic_node(state):
        answer = state["final_answer"]

        # ===== 🔥 STEP 1: CHECK IF IMPROVEMENT IS NEEDED (LLM-BASED) =====
        def needs_improvement(answer):
            prompt = f"""
            Decide whether this answer needs improvement.

            Answer:
            {answer}

            Evaluate based on:
            - clarity
            - completeness
            - usefulness to the user

            Return YES if:
            - answer is unclear
            - answer is incomplete
            - answer is poorly structured
            - answer is not helpful

            Return NO if:
            - answer is clear
            - answer is complete
            - answer is useful and well-structured

            IMPORTANT:
            - Even short answers can be GOOD
            - Even long answers can be BAD

            Return ONLY YES or NO.
            """

            decision = safe_llm_call(prompt)
            decision_clean = decision.strip().lower()

            print("🧠 Critic decision:", decision_clean)

            return "yes" in decision_clean

        # ===== 🔥 STEP 2: DECIDE =====
        improve = needs_improvement(answer)

        if not improve:
            print("✅ Critic skipped (answer is good)")

            # 🔥 OPTIONAL SAFETY (ADD THIS)
            state["metrics"] = state.get("metrics", {})

            return state

        # ===== 🔥 STEP 3: IMPROVE ANSWER =====
        prompt = f"""
        Improve this answer ONLY if it is unclear or incomplete.

        Answer:
        {answer}

        STRICT RULES:
        - Do NOT change meaning
        - Do NOT add new information
        - Do NOT hallucinate
        - Only improve clarity, structure, and readability
        - Keep it concise and clean
        - If already good → return as-is

        OUTPUT:
        Improved answer only
        """

        improved = safe_llm_call(prompt)

        # ===== 🔥 STEP 4: SAFETY CHECK =====
        if improved:
            improved_clean = improved.strip()

            # ensure meaningful improvement
            if (
                    len(improved_clean) > 0
                    and improved_clean != answer.strip()
            ):
                print("✨ Critic improved answer")

                state["final_answer"] = improved_clean

                # preserve metrics
                state["confidence"] = state.get("confidence")
                state["retrieval_count"] = state.get("retrieval_count")
                state["context_length"] = state.get("context_length")
                state["answer_length"] = len(improved_clean)

                state["metrics"] = {
                    "confidence": state.get("confidence", 0.0),
                    "context_length": state.get("context_length", 0),
                    "context_quality": state.get("context_quality", 0.0),
                    "final_score": state.get("final_score", 0.0),
                    "answer_length": len(state.get("final_answer", "")),
                    "retrieval_count": state.get("retrieval_count", 0)
                }

            else:
                print("⚠️ Critic skipped (no meaningful improvement)")
        else:
            print("⚠️ Critic failed (empty response)")

        return state

    # ===== FOLLOW-UP =====
    @traceable(name="followup")
    def followup_node(state):
        answer = state["final_answer"]


        if state.get("needs_fallback"):
            print("⛔ Skipping followups (fallback needed)")
            return state

        def is_answer_useful(answer):
            prompt = f"""
            Is this answer useful enough to suggest follow-up questions?

            Answer:
            {answer}

            Return YES or NO.
            """
            decision = safe_llm_call(prompt).strip().lower()

            print("🧠 Followup usefulness:", decision)

            return "yes" in decision

        if not is_answer_useful(answer):
            print("⛔ Skipping followups (answer not useful)")
            return state

        prompt = f"""
    Suggest 2 short and relevant follow-up questions.

    Answer:
    {answer}

    Rules:
    - Keep questions concise
    - Make them useful for deeper understanding
    - Do NOT repeat the same question
    - Format exactly:

    1. ...
    2. ...
    """

        followups = safe_llm_call(prompt)

        state["final_answer"] += f"\n\n💡 Follow-up questions:\n{followups}"
        # 🔥 PRESERVE METRICS
        state["confidence"] = state.get("confidence")
        state["retrieval_count"] = state.get("retrieval_count")
        state["context_length"] = state.get("context_length")
        state["answer_length"] = len(state["final_answer"])

        state["metrics"] = {
            "confidence": state.get("confidence", 0.0),
            "context_length": state.get("context_length", 0),
            "context_quality": state.get("context_quality", 0.0),
            "final_score": state.get("final_score", 0.0),
            "answer_length": len(state.get("final_answer", "")),
            "retrieval_count": state.get("retrieval_count", 0)
        }

        return state

    # ===== GRAPH FLOW =====
    graph.add_node("planner", planner)
    graph.add_node("tool", tool)
    graph.add_node("reasoning_node", reasoning_node)
    graph.add_node("memory_node", memory_node)
    graph.add_node("answer_node", answer_node)
    graph.add_node("fallback_node", fallback_node)
    graph.add_node("critic_node", critic_node)

    # ✅ ADD THIS
    graph.add_node("followup_node", followup_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "tool")
    graph.add_edge("tool", "reasoning_node")
    graph.add_edge("reasoning_node", "memory_node")
    graph.add_edge("memory_node", "answer_node")
    graph.add_edge("answer_node", "fallback_node")
    graph.add_edge("fallback_node", "critic_node")

    # 🔥 ONLY CHANGE HERE
    graph.add_edge("critic_node", "followup_node")
    graph.add_edge("followup_node", END)

    return graph.compile()
# ===== API =====
@traceable(name="youtube-agent")
@app.post("/api/ask-stream")
async def ask_stream(request: Request):
    try:
        data = await request.json()

        video_url = data.get("video_url")
        question = data.get("question")

        # 🔥 VALIDATION
        if not video_url or not question:
            return {"error": "Missing video_url or question"}

        session_id = data.get("session_id", "default")

        # 🔥 EXTRACT VIDEO ID
        video_id = extract_video_id(video_url)
        if not video_id:
            return {"error": "Invalid YouTube URL"}

        # 🔥 MAKE USER+VIDEO UNIQUE (IMPORTANT FIX)
        user_id = session_id

        # ===== VECTOR STORE =====
        store = get_or_create_vectorstore(video_url)

        # ===== TRANSCRIPT =====
        # ===== TRANSCRIPT =====
        transcript = load_transcript(video_id)

        if not transcript:
            print("⚠️ Loading transcript from Apify")

            transcript, _ = fetch_transcript_apify(video_url)

            if transcript:
                save_transcript(video_id, transcript)

        if not transcript:
            print("❌ No transcript available at all")
            transcript = "Transcript not available."

        # ===== BUILD GRAPH =====
        graph = build_graph(store, user_id, transcript)

        # ===== RUN AGENT =====
        result = graph.invoke({
            "question": question,
            "plan": "",
            "tool_output": "",
            "memory_context": "",
            "final_answer": ""
        })

        final_answer = result["final_answer"]

        # 🔥 GET METRICS DIRECTLY FROM STATE
        metrics = result.get("metrics", {})

        # 🔥 SAFE FALLBACK (if missing)
        metrics = {
            "confidence": metrics.get("confidence", result.get("confidence", 0.0)),
            "context_length": metrics.get("context_length", result.get("context_length", 0)),
            "context_quality": metrics.get("context_quality", result.get("context_quality", 0.0)),
            "final_score": metrics.get("final_score", result.get("final_score", 0.0)),
            "answer_length": metrics.get("answer_length", len(result.get("final_answer", ""))),
            "retrieval_count": metrics.get("retrieval_count", result.get("retrieval_count", 0)),
            "source": "Wikipedia" if "Wikipedia" in result.get("final_answer", "") else "Transcript"
        }

        print("\n📊 ===== AI METRICS =====")
        print("Confidence:", metrics["confidence"])
        print("Retrieval Count:", metrics["retrieval_count"])
        print("Context Length:", metrics["context_length"])
        print("Context Quality:", metrics["context_quality"])
        print("Final Score:", metrics["final_score"])
        print("Answer Length:", metrics["answer_length"])
        print("Source:", metrics["source"])
        print("========================\n")
        # ===== SAVE MEMORY =====
        save_memory(user_id, question, final_answer)
        save_chat_history(user_id, question, final_answer)

        # ===== ✅ CORRECT STREAMING (FINAL FIX) =====
        def stream():
            try:
                # 🔹 send answer
                yield f"data: {final_answer}\n\n"

                # 🔹 send metrics (VERY IMPORTANT)
                yield f"event: metrics\ndata: {json.dumps(metrics)}\n\n"

            except Exception as e:
                yield f"data: Error\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/api/history/{session_id}")
def get_history(session_id: str):
    history = load_chat_history(session_id)
    return {"history": history}

@app.delete("/api/history/{session_id}")
def delete_history(session_id: str):
    try:
        print(f"🗑️ Deleting history for session: {session_id}")

        bucket = get_bucket()
        blob = bucket.blob(f"chat_history/{session_id}.json")

        if blob.exists():
            blob.delete()
            print("✅ History deleted from GCS")
        else:
            print("⚠️ No history found in GCS")

        return {"status": "deleted"}

    except Exception as e:
        print("❌ Delete error:", e)
        return {"error": str(e)}

# ===== LOCAL =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)