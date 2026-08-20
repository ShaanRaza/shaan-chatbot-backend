"""
rag_engine.py — BM25 RAG engine for Shaan Raza AI chatbot.
Handles knowledge base loading, recursive chunking, indexing, and retrieval.

Retrieval is pure BM25 (rank_bm25.BM25Okapi) — lexical/keyword search,
in-memory, no embedding model or vector database. Chosen for low memory
footprint (fits comfortably on free-tier hosting) and low latency.
"""

import os
import re
import json
import numpy as np
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi


# ─────────────────────────────────────────────────────────────────
# Chunk Schema
# ─────────────────────────────────────────────────────────────────
# Each chunk: {
#   "id": str,
#   "source": str,         # "resume" | "github" | "project" | "calendar"
#   "section": str,        # Human-readable section name
#   "content": str,        # Searchable text content
#   "metadata": dict       # Extra context (repo URL, dates, etc.)
# }

RESUME_FILE = "resume.txt"
GITHUB_FILE = "knowledge/github_repos.json"
CALENDAR_FILE = "calendar_store.json"

RECURSIVE_CHUNK_SIZE = 800   # target chars per chunk
RECURSIVE_CHUNK_OVERLAP = 120


# ─────────────────────────────────────────────────────────────────
# Recursive chunking (structural: paragraphs → lines → sentences →
# words → chars, no embedding model required)
# ─────────────────────────────────────────────────────────────────

def recursive_chunk_text(text: str, chunk_size: int = RECURSIVE_CHUNK_SIZE,
                          chunk_overlap: int = RECURSIVE_CHUNK_OVERLAP,
                          separators: Optional[List[str]] = None) -> List[str]:
    """Split text into overlapping chunks, preferring larger structural
    boundaries (paragraph > line > sentence > word) before falling back
    to a hard character split."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    separator = separators[0]
    remaining = separators[1:]
    pieces = list(text) if separator == "" else text.split(separator)

    # Recurse into any piece that's still too large before merging.
    expanded = []
    for piece in pieces:
        if len(piece) > chunk_size and remaining:
            expanded.extend(recursive_chunk_text(piece, chunk_size, chunk_overlap, remaining))
        elif piece:
            expanded.append(piece)

    return _merge_pieces(expanded, separator, chunk_size, chunk_overlap)


def _merge_pieces(pieces: List[str], separator: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Greedily pack small pieces back together up to chunk_size, carrying
    a trailing overlap window into the next chunk for context continuity."""
    chunks = []
    current: List[str] = []
    current_len = 0
    sep_len = len(separator)

    def flush():
        if current:
            chunks.append(separator.join(current).strip())

    for piece in pieces:
        piece_len = len(piece)
        added_len = piece_len + (sep_len if current else 0)
        if current and current_len + added_len > chunk_size:
            flush()
            # Keep trailing pieces (by char budget) for overlap continuity.
            overlap_pieces = []
            overlap_len = 0
            for p in reversed(current):
                if overlap_len + len(p) > chunk_overlap:
                    break
                overlap_pieces.insert(0, p)
                overlap_len += len(p) + sep_len
            current = overlap_pieces
            current_len = overlap_len

        current.append(piece)
        current_len += piece_len + (sep_len if len(current) > 1 else 0)

    flush()
    return [c for c in chunks if c]


def _tokenize(text: str) -> List[str]:
    """Simple lowercase word tokenizer for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


class RAGEngine:
    def __init__(self):
        self.chunks: List[Dict] = []
        self.is_loaded = False

        self.bm25: Optional[BM25Okapi] = None
        self._id_to_index: Dict[str, int] = {}

    # ─────────────────────────────────────────────────────────────
    # Knowledge Base Loading
    # ─────────────────────────────────────────────────────────────

    def load(self):
        """Load all knowledge sources and build the BM25 + semantic index."""
        try:
            chunks = []

            try:
                chunks.extend(self._load_resume_chunks())
            except Exception as e:
                print(f"[ERROR] Failed to load resume chunks: {e}")
                import traceback
                traceback.print_exc()
                raise e

            try:
                chunks.extend(self._load_github_chunks())
            except Exception as e:
                print(f"[ERROR] Failed to load GitHub chunks: {e}")
                import traceback
                traceback.print_exc()
                raise e

            try:
                chunks.extend(self._load_calendar_chunks())
            except Exception as e:
                print(f"[ERROR] Failed to load calendar chunks: {e}")
                import traceback
                traceback.print_exc()
                raise e

            self.chunks = chunks
            self._build_index()
            self.is_loaded = True
            print(f"[RAG] Knowledge base loaded successfully: {len(chunks)} chunks")
            self._print_chunk_summary()
        except Exception as e:
            self.is_loaded = False
            print(f"[ERROR] Failed to load knowledge base: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def _print_chunk_summary(self):
        by_source = {}
        for c in self.chunks:
            src = c["source"]
            by_source[src] = by_source.get(src, 0) + 1
        for src, count in by_source.items():
            print(f"[RAG]   {src}: {count} chunks")

    # ─────────────────────────────────────────────────────────────
    # Resume Chunking — Section-aware + recursive sub-chunking
    # ─────────────────────────────────────────────────────────────

    def _load_resume_chunks(self) -> List[Dict]:
        chunks = []
        if not os.path.exists(RESUME_FILE):
            print(f"[RAG] WARNING: {RESUME_FILE} not found")
            return chunks

        with open(RESUME_FILE, "r") as f:
            content = f.read()

        header_match = re.search(r"^(.*?)(?=={3,})", content, re.DOTALL)
        if header_match:
            header = header_match.group(1).strip()
            chunks.append(self._make_chunk(
                "resume_header",
                "resume",
                "Personal Information & Contact",
                f"Shaan Raza — Personal Information\n{header}",
                {"file": RESUME_FILE}
            ))

        sections = re.split(r"={3,}\n\d+\.\s+", content)
        section_headers = re.findall(r"={3,}\n\d+\.\s+([^\n]+)", content)

        for i, (header, body) in enumerate(zip(section_headers, sections[1:])):
            header = header.strip()
            body = body.strip()

            parts = recursive_chunk_text(body)
            if len(parts) == 1:
                chunks.append(self._make_chunk(
                    f"resume_section_{i}",
                    "resume",
                    header,
                    f"Section: {header}\n{parts[0]}",
                    {"file": RESUME_FILE, "section_index": i}
                ))
            else:
                for part_idx, part_text in enumerate(parts):
                    chunks.append(self._make_chunk(
                        f"resume_section_{i}_part{part_idx}",
                        "resume",
                        header,
                        f"Section: {header} (Part {part_idx + 1})\n{part_text}",
                        {"file": RESUME_FILE, "section_index": i, "part": part_idx}
                    ))

        chunks.append(self._make_chunk(
            "resume_rolefit",
            "resume",
            "Role Fit & Why Hire Shaan",
            """Why should you hire Shaan Raza?

Shaan Raza is a Data Analyst and AI Developer with hands-on experience spanning:
- Environmental sustainability data analytics at Carbon Crunch (GHG emissions, Scope 1-2-3, LCA)
- Business process automation at Crystal Technology Services (IVR, voicebot, workflow automation for HDFC ERGO, Boat, Samsung)
- Sales analytics at Pregrad (generated INR 50,000+ in revenue, analyzed 1000+ customers)
- End-to-end ML pipeline: FMCG churn prediction with XGBoost (ROC-AUC 0.92+)
- SQL expertise: 150+ problems solved across LeetCode, HackerRank, DataLemur
- Microsoft Power BI PL-300 Certified
- BTech Electronics & Communication Engineering, Jamia Millia Islamia (CGPA 8.2/10)
- 3rd place among 380+ in Jamia Case Challenge

Shaan's combination of technical depth (ML, SQL, Python automation) with business analysis skills (BRDs, FRDs, stakeholder management) makes him an exceptional candidate for data-driven roles.""",
            {"file": RESUME_FILE}
        ))

        return chunks

    # ─────────────────────────────────────────────────────────────
    # GitHub Chunks — recursive sub-chunking for long content
    # ─────────────────────────────────────────────────────────────

    def _load_github_chunks(self) -> List[Dict]:
        chunks = []

        if not os.path.exists(GITHUB_FILE) or os.environ.get("REFRESH_GITHUB_CACHE") == "true":
            try:
                print("[RAG] Fetching latest GitHub repos dynamically...")
                from github_fetcher import fetch_all_repos, GITHUB_USERNAME, KNOWN_REPOS, build_rag_content
                repos = fetch_all_repos(GITHUB_USERNAME, list(KNOWN_REPOS))

                # Merge per-repo: a repo that failed/rate-limited this run keeps its
                # previous cached entry instead of being overwritten with degraded data.
                old_by_name = {}
                if os.path.exists(GITHUB_FILE):
                    with open(GITHUB_FILE, "r") as f:
                        old_by_name = {r.get("name"): r for r in json.load(f)}

                merged = []
                for repo in repos:
                    name = repo.get("name")
                    if not repo.get("fetched_from_api") and name in old_by_name:
                        print(f"[RAG]   Keeping cached data for {name} (rate-limited/failed this run)")
                        merged.append(old_by_name[name])
                        continue
                    repo["rag_content"] = build_rag_content(repo)
                    merged.append(repo)

                os.makedirs("knowledge", exist_ok=True)
                with open(GITHUB_FILE, "w") as f:
                    json.dump(merged, f, indent=2)
                print("[RAG] Successfully fetched and cached live GitHub data.")
            except Exception as e:
                print(f"[RAG] WARNING: Failed to fetch live GitHub data, using local cache or fallback: {e}")
        else:
            print("[RAG] Using cached GitHub data from local storage.")

        if not os.path.exists(GITHUB_FILE):
            print(f"[RAG] WARNING: {GITHUB_FILE} not found. Run github_fetcher.py first.")
            return self._get_github_fallback_chunks()

        with open(GITHUB_FILE, "r") as f:
            repos = json.load(f)

        for repo in repos:
            name = repo.get("name", "Unknown")
            rag_content = repo.get("rag_content", "")
            readme_summary = repo.get("readme_summary", "")
            readme = repo.get("readme", "")
            meta = {
                "repo_name": name,
                "url": repo.get("url", f"https://github.com/ShaanRaza/{name}"),
                "language": repo.get("language", ""),
                "topics": repo.get("topics", [])
            }

            primary_content = rag_content or readme_summary or readme[:2000]
            if primary_content:
                parts = recursive_chunk_text(primary_content)
                for part_idx, part_text in enumerate(parts):
                    suffix = f"_part{part_idx}" if len(parts) > 1 else ""
                    chunks.append(self._make_chunk(
                        f"github_{name}{suffix}",
                        "github",
                        f"GitHub Repository: {name}",
                        part_text,
                        meta
                    ))

            if len(readme) > 1000:
                readme_parts = recursive_chunk_text(readme[:4000])
                for part_idx, part_text in enumerate(readme_parts):
                    chunks.append(self._make_chunk(
                        f"github_{name}_readme_part{part_idx}",
                        "github",
                        f"GitHub Repository README: {name}",
                        f"README for {name}:\n{part_text}",
                        {"repo_name": name}
                    ))

        print(f"[RAG] Loaded {len(chunks)} GitHub chunks for {len(repos)} repos")
        return chunks

    def _get_github_fallback_chunks(self) -> List[Dict]:
        return [
            self._make_chunk(
                "github_overview",
                "github",
                "GitHub Repositories Overview",
                """Shaan Raza's GitHub repositories (github.com/ShaanRaza):
1. Zomato_Dataset_Analysis — SQL + Pandas market analysis for restaurant expansion opportunities
2. Automation-Code-for-RTDMS — Selenium + BeautifulSoup scraper for India's environmental monitoring portal
3. EPD-Models-openLCA — LCA models using EXIOBASE/Ecoinvent for ISO-compliant Environmental Product Declarations
4. PowerBI-Dashboards — Collection of interactive Power BI dashboards (emissions, sales, operations)
5. FMCG-Customer-Churn-Prediction — XGBoost churn model with ROC-AUC 0.92+, RFM analysis
6. Case-Competitions — Business case portfolio; 3rd place in Jamia Case Challenge (380+ participants)""",
                {}
            )
        ]

    # ─────────────────────────────────────────────────────────────
    # Calendar Chunks
    # ─────────────────────────────────────────────────────────────

    def _load_calendar_chunks(self) -> List[Dict]:
        chunks = []
        calendar_paths = [CALENDAR_FILE, "../voice-agent-interview/calendar_store.json"]
        calendar_data = None

        for path in calendar_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        calendar_data = json.load(f)
                    break
                except Exception:
                    pass

        if calendar_data:
            available_slots = [s for s in calendar_data if s.get("status") == "available"]
            if available_slots:
                slot_lines = []
                grouped = {}
                for s in available_slots:
                    grouped.setdefault(s["date"], []).append(s["time"])

                for date, times in sorted(grouped.items())[:7]:
                    slot_lines.append(f"  {date}: {', '.join(times)}")

                chunks.append(self._make_chunk(
                    "calendar_availability",
                    "calendar",
                    "Interview Availability & Booking Info",
                    "Shaan Raza's available interview slots:\n" + "\n".join(slot_lines) +
                    "\n\nInterview Details:"
                    "\n- Platform: Google Meet (video call link sent after booking confirmation)"
                    "\n- Duration: 1 hour per session"
                    "\n- Timezone: IST (India Standard Time, UTC+5:30)"
                    "\n- International conversions: IST 9AM = PST 7:30PM prev day | EST 10:30PM prev day | GMT 3:30AM"
                    "\n- To book: provide your full name, email, and preferred date + time slot"
                    "\n- To cancel or reschedule: provide your email and booked slot"
                    "\n- Available Monday to Friday only (no weekends)",
                    {"source_file": "calendar_store.json"}
                ))
        else:
            chunks.append(self._make_chunk(
                "calendar_availability",
                "calendar",
                "Interview Availability & Booking Info",
                "Shaan Raza is available for interviews Monday to Friday."
                " Platform: Google Meet (video link sent on confirmation)."
                " Duration: 1 hour. Timezone: IST (UTC+5:30)."
                " IST 9AM = PST 7:30PM prev day | EST 10:30PM prev day | GMT 3:30AM."
                " To book provide your name, email, and preferred slot."
                " To cancel or reschedule provide your email and booked slot.",
                {}
            ))

        return chunks

    # ─────────────────────────────────────────────────────────────
    # BM25 Index — pure lexical, in-memory, no embeddings/vector DB
    # ─────────────────────────────────────────────────────────────

    def _build_index(self):
        texts = [c["content"] for c in self.chunks]
        ids = [c["id"] for c in self.chunks]
        self._id_to_index = {cid: i for i, cid in enumerate(ids)}

        tokenized_corpus = [_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[RAG] BM25 index built over {len(texts)} chunks")

    # ─────────────────────────────────────────────────────────────
    # Retrieval — BM25 ranking
    # ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 6, threshold: float = 0.0, force_calendar: bool = False) -> List[Dict]:
        """Retrieve top-k relevant chunks for a query via BM25."""
        if not self.is_loaded:
            raise RuntimeError("Knowledge base is not loaded. Ensure synchronous startup load succeeded.")

        query_lower = query.lower()
        if any(w in query_lower for w in ["your", "this repo", "this project", "this chatbot", "chatbot repository", "chatbot backend", "chatbot code", "you change"]):
            query = query + " shaan-chatbot-backend"

        pool_size = min(top_k * 4, len(self.chunks))

        bm25_scores = self.bm25.get_scores(_tokenize(query))
        ranked_idx = np.argsort(bm25_scores)[::-1][:pool_size]

        results = []
        seen_sources = set()
        for idx in ranked_idx:
            score = float(bm25_scores[idx])
            if score <= threshold:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = round(score, 4)

            section_key = f"{chunk['source']}::{chunk['section']}"
            if section_key in seen_sources:
                continue
            seen_sources.add(section_key)

            results.append(chunk)
            if len(results) >= top_k:
                break

        if not results:
            fallback_ids = ["resume_header", "resume_rolefit"]
            for chunk in self.chunks:
                if chunk["id"] in fallback_ids:
                    c = chunk.copy()
                    c["score"] = 0.0
                    results.append(c)

        if force_calendar:
            already_has_calendar = any(c.get("source") == "calendar" for c in results)
            if not already_has_calendar:
                for chunk in self.chunks:
                    if chunk.get("source") == "calendar":
                        cal = chunk.copy()
                        cal["score"] = max((c.get("score", 0) for c in results), default=0) + 0.01
                        results.append(cal)
                        break

        return results

    def retrieve_and_build_context(self, query: str, top_k: int = 6, force_calendar: bool = False) -> tuple:
        chunks = self.retrieve(query, top_k=top_k, force_calendar=force_calendar)

        if not chunks:
            return "", []

        context_parts = []
        sources = []

        for i, chunk in enumerate(chunks, 1):
            source_label = self._format_source_label(chunk)
            context_parts.append(f"[SOURCE {i}: {source_label}]\n{chunk['content']}")
            sources.append({
                "label": source_label,
                "source": chunk["source"],
                "section": chunk["section"],
                "score": round(chunk.get("score", 0), 4),
                "metadata": chunk.get("metadata", {}),
                "content": chunk["content"]
            })

        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    def _format_source_label(self, chunk: Dict) -> str:
        src = chunk["source"]
        section = chunk["section"]
        if src == "github":
            meta = chunk.get("metadata", {})
            repo = meta.get("repo_name", section.replace("GitHub Repository: ", ""))
            return f"GitHub/{repo}"
        elif src == "resume":
            return f"Resume/{section}"
        elif src == "calendar":
            return "Calendar/Availability"
        return f"{src}/{section}"

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _make_chunk(chunk_id: str, source: str, section: str, content: str, metadata: dict) -> Dict:
        return {
            "id": chunk_id,
            "source": source,
            "section": section,
            "content": content,
            "metadata": metadata
        }

    def get_stats(self) -> Dict:
        by_source = {}
        for c in self.chunks:
            by_source[c["source"]] = by_source.get(c["source"], 0) + 1
        return {
            "total_chunks": len(self.chunks),
            "by_source": by_source,
            "retrieval": "BM25",
            "embedding_model": None,
            "vector_store": None,
            "is_loaded": self.is_loaded
        }
