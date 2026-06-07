"""
rag_engine.py — TF-IDF based RAG engine for Shaan Raza AI chatbot.
Handles knowledge base loading, chunking, indexing, and retrieval.
"""

import os
import json
import re
import numpy as np
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


class RAGEngine:
    def __init__(self):
        self.chunks: List[Dict] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.chunk_matrix = None
        self.is_loaded = False

    # ─────────────────────────────────────────────────────────────
    # Knowledge Base Loading
    # ─────────────────────────────────────────────────────────────

    def load(self):
        """Load all knowledge sources and build the TF-IDF index."""
        try:
            chunks = []
            
            # Load resume chunks
            try:
                resume_chunks = self._load_resume_chunks()
                chunks.extend(resume_chunks)
            except Exception as e:
                print(f"[ERROR] Failed to load resume chunks: {e}")
                import traceback
                traceback.print_exc()
                raise e
                
            # Load GitHub chunks
            try:
                github_chunks = self._load_github_chunks()
                chunks.extend(github_chunks)
            except Exception as e:
                print(f"[ERROR] Failed to load GitHub chunks: {e}")
                import traceback
                traceback.print_exc()
                raise e
                
            # Load Calendar chunks
            try:
                calendar_chunks = self._load_calendar_chunks()
                chunks.extend(calendar_chunks)
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
    # Resume Chunking — Section-aware
    # ─────────────────────────────────────────────────────────────

    def _load_resume_chunks(self) -> List[Dict]:
        chunks = []
        if not os.path.exists(RESUME_FILE):
            print(f"[RAG] WARNING: {RESUME_FILE} not found")
            return chunks

        with open(RESUME_FILE, "r") as f:
            content = f.read()

        # Header chunk (contact info)
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

        # Split by numbered sections (===...===)
        sections = re.split(r"={3,}\n\d+\.\s+", content)
        section_headers = re.findall(r"={3,}\n\d+\.\s+([^\n]+)", content)

        for i, (header, body) in enumerate(zip(section_headers, sections[1:])):
            header = header.strip()
            body = body.strip()

            # Split long sections into sub-chunks
            if len(body) > 800:
                sub_chunks = self._split_section(body, header, i)
                chunks.extend(sub_chunks)
            else:
                chunks.append(self._make_chunk(
                    f"resume_section_{i}",
                    "resume",
                    header,
                    f"Section: {header}\n{body}",
                    {"file": RESUME_FILE, "section_index": i}
                ))

        # Add a combined "role fit" chunk for quick hire-me queries
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

    def _split_section(self, text: str, section_name: str, section_idx: int) -> List[Dict]:
        """Split a long section into overlapping chunks of ~300 words."""
        words = text.split()
        chunks = []
        chunk_size = 200
        overlap = 40
        i = 0
        part = 0

        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append(self._make_chunk(
                f"resume_section_{section_idx}_part{part}",
                "resume",
                section_name,
                f"Section: {section_name} (Part {part + 1})\n{chunk_text}",
                {"section_index": section_idx, "part": part}
            ))
            i += chunk_size - overlap
            part += 1

        return chunks

    # ─────────────────────────────────────────────────────────────
    # GitHub Chunks
    # ─────────────────────────────────────────────────────────────

    def _load_github_chunks(self) -> List[Dict]:
        chunks = []
        
        # Try to fetch fresh GitHub data dynamically
        try:
            print("[RAG] Fetching latest GitHub repos dynamically...")
            from github_fetcher import fetch_all_repos, GITHUB_USERNAME, KNOWN_REPOS, build_rag_content
            repos = fetch_all_repos(GITHUB_USERNAME, list(KNOWN_REPOS))
            for repo in repos:
                repo["rag_content"] = build_rag_content(repo)
            os.makedirs("knowledge", exist_ok=True)
            with open(GITHUB_FILE, "w") as f:
                json.dump(repos, f, indent=2)
            print("[RAG] Successfully fetched and cached live GitHub data.")
        except Exception as e:
            print(f"[RAG] WARNING: Failed to fetch live GitHub data, using local cache or fallback: {e}")

        if not os.path.exists(GITHUB_FILE):
            print(f"[RAG] WARNING: {GITHUB_FILE} not found. Run github_fetcher.py first.")
            # Use fallback minimal chunks
            return self._get_github_fallback_chunks()

        with open(GITHUB_FILE, "r") as f:
            repos = json.load(f)

        for repo in repos:
            name = repo.get("name", "Unknown")
            rag_content = repo.get("rag_content", "")
            readme_summary = repo.get("readme_summary", "")
            readme = repo.get("readme", "")

            # Primary chunk — full RAG content
            primary_content = rag_content or readme_summary or readme[:2000]
            if primary_content:
                chunks.append(self._make_chunk(
                    f"github_{name}",
                    "github",
                    f"GitHub Repository: {name}",
                    primary_content,
                    {
                        "repo_name": name,
                        "url": repo.get("url", f"https://github.com/ShaanRaza/{name}"),
                        "language": repo.get("language", ""),
                        "topics": repo.get("topics", [])
                    }
                ))

            # If README is long, add a second chunk
            if len(readme) > 1000:
                chunks.append(self._make_chunk(
                    f"github_{name}_readme",
                    "github",
                    f"GitHub Repository README: {name}",
                    f"README for {name}:\n{readme[:3000]}",
                    {"repo_name": name}
                ))

        print(f"[RAG] Loaded {len(chunks)} GitHub chunks for {len(repos)} repos")
        return chunks

    def _get_github_fallback_chunks(self) -> List[Dict]:
        """Minimal fallback chunks if github_repos.json doesn't exist yet."""
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

        # Try to load from the voice-agent calendar or local one
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
            # Static fallback availability note
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
    # TF-IDF Index
    # ─────────────────────────────────────────────────────────────

    def _build_index(self):
        """Build TF-IDF matrix over all chunk contents."""
        texts = [c["content"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )
        self.chunk_matrix = self.vectorizer.fit_transform(texts)
        print(f"[RAG] TF-IDF index built: {self.chunk_matrix.shape}")

    # ─────────────────────────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 6, threshold: float = 0.03) -> List[Dict]:
        """Retrieve top-k relevant chunks for a query."""
        if not self.is_loaded:
            raise RuntimeError("Knowledge base is not loaded. Ensure synchronous startup load succeeded.")

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.chunk_matrix)[0]

        top_indices = np.argsort(scores)[::-1][:top_k * 2]

        results = []
        seen_sources = set()

        for idx in top_indices:
            score = float(scores[idx])
            if score < threshold:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = score

            # Deduplicate by section (avoid two chunks from the same section)
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
                    c["score"] = 0.01
                    results.append(c)

        return results

    def retrieve_and_build_context(self, query: str, top_k: int = 6) -> tuple:
        """Retrieve chunks and build a formatted context string with citations."""
        chunks = self.retrieve(query, top_k=top_k)

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
                "score": round(chunk.get("score", 0), 3),
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
            "index_shape": list(self.chunk_matrix.shape) if self.chunk_matrix is not None else None,
            "is_loaded": self.is_loaded
        }
