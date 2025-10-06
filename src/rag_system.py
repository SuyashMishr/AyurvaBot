"""
RAG (Retrieval-Augmented Generation) System for Enhanced AyurvaBot
Handles vector search, embeddings, and AI model integration.
"""

import os
import json
import hashlib
import numpy as np
try:
    import faiss
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from transformers import pipeline
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("⚠️ Some dependencies not available. Running in fallback mode.")

from .knowledge_base import AyurvedaKnowledgeBase
from .query_processor import QueryProcessor
from .entity_extractor import EntityExtractor
from .hybrid_retriever import HybridRetriever
from .openrouter_client import OpenRouterClient

class RAGSystem:
    """
    RAG system for intelligent document retrieval and response generation.
    """
    
    def __init__(self, config):
        self.config = config
        self.sentence_transformer = None
        self.cross_encoder = None  # actual CrossEncoder for reranking
        self.hf_pipeline = None  # legacy QA
        self.faiss_index = None
        self.hybrid_retriever = None
        self.openrouter_client = OpenRouterClient(api_key=config.OPENROUTER_API_KEY, default_model=config.OPENROUTER_MODEL)
        self.query_processor = QueryProcessor()
        self.entity_extractor = EntityExtractor()
        self.knowledge_base = AyurvedaKnowledgeBase(config=config)
        self.document_chunks = []
        
        # Set Hugging Face token
        if DEPENDENCIES_AVAILABLE:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = config.HF_TOKEN
    
    def load_knowledge_base(self, _config=None):  # accept optional param for backward call compatibility
        """Load Ayurvedic knowledge base (dataset ingestion + curated)."""
        try:
            chunk_count = self.knowledge_base.load_knowledge_chunks(use_csv=False, use_synthetic_json=True, config=self.config)
            self.document_chunks = self.knowledge_base.get_document_chunks()
            print(f"✅ Loaded {chunk_count} knowledge chunks")
            return True
        except Exception as e:
            print(f"❌ Error loading knowledge base: {e}")
            return False
    
    def initialize_sentence_transformer(self):
        """
        Initialize the sentence transformer for embeddings.
        """
        if not DEPENDENCIES_AVAILABLE:
            print("⚠️ Sentence Transformer not available. Using fallback mode.")
            return False
            
        try:
            print("Loading Sentence Transformer for RAG embeddings...")
            self.sentence_transformer = SentenceTransformer(self.config.EMBEDDING_MODEL)
            # Attempt proper CrossEncoder load (optional reranker)
            try:
                self.cross_encoder = CrossEncoder(self.config.CROSS_ENCODER_MODEL)
                print("✅ CrossEncoder model loaded for reranking")
            except Exception as ce:
                print(f"⚠️ CrossEncoder load failed (will fallback): {ce}")
            print("✅ Sentence Transformer loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading Sentence Transformer: {e}")
            return False
    
    def _compute_corpus_fingerprint(self, first_n: int = 200) -> str:
        hasher = hashlib.sha256()
        for text in self.knowledge_base.get_knowledge_chunks()[:first_n]:
            hasher.update(text[:500].encode('utf-8', errors='ignore'))
        hasher.update(self.config.EMBEDDING_MODEL.encode())
        return hasher.hexdigest()

    def _load_existing_index(self) -> bool:
        meta_path = self.config.INDEX_METADATA_PATH
        if not (os.path.exists(self.config.FAISS_INDEX_PATH) and os.path.exists(self.config.EMBEDDINGS_CACHE_PATH) and os.path.exists(meta_path)):
            return False
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            current_fp = self._compute_corpus_fingerprint()
            if meta.get('fingerprint') != current_fp:
                print("ℹ️ Index fingerprint mismatch; rebuilding.")
                return False
            # load embeddings (optional)
            embeddings = np.load(self.config.EMBEDDINGS_CACHE_PATH)
            # load faiss
            self.faiss_index = faiss.read_index(self.config.FAISS_INDEX_PATH)
            if self.faiss_index.ntotal != embeddings.shape[0]:
                print("ℹ️ FAISS vector count mismatch; rebuilding.")
                return False
            print(f"✅ Loaded persisted FAISS index ({self.faiss_index.ntotal} vectors)")
            return True
        except Exception as e:
            print(f"⚠️ Failed loading existing index: {e}")
            return False

    def _save_index(self, embeddings: np.ndarray, fingerprint: str):
        try:
            faiss.write_index(self.faiss_index, self.config.FAISS_INDEX_PATH)
            np.save(self.config.EMBEDDINGS_CACHE_PATH, embeddings)
            meta = {
                'fingerprint': fingerprint,
                'model': self.config.EMBEDDING_MODEL,
                'vectors': int(self.faiss_index.ntotal),
                'dimension': int(embeddings.shape[1]),
            }
            with open(self.config.INDEX_METADATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)
            print("💾 FAISS index & metadata saved.")
        except Exception as e:
            print(f"⚠️ Failed saving index: {e}")

    def _encode_in_batches(self, texts, batch_size):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            emb = self.sentence_transformer.encode(batch, show_progress_bar=False)
            all_embeddings.append(emb)
        return np.vstack(all_embeddings)

    def initialize_faiss_index(self, force_rebuild: bool = False):
        """Initialize or load FAISS index (persistent)."""
        if not DEPENDENCIES_AVAILABLE or not self.sentence_transformer or not self.knowledge_base.get_knowledge_chunks():
            print("⚠️ Cannot initialize FAISS: missing dependencies or data")
            return False
        # Try load existing
        if not force_rebuild and self._load_existing_index():
            # Prepare hybrid retriever
            if self.config.USE_HYBRID_RETRIEVAL:
                self.hybrid_retriever = HybridRetriever(
                    faiss_index=self.faiss_index,
                    embedding_model=self.sentence_transformer,
                    document_chunks=self.document_chunks,
                    query_processor=self.query_processor,
                    entity_extractor=self.entity_extractor,
                    cross_encoder=self.cross_encoder,
                    similarity_threshold=self.config.SIMILARITY_THRESHOLD,
                )
                print("✅ Hybrid retriever ready (loaded index)")
            return True
        # Build new
        try:
            knowledge_chunks = self.knowledge_base.get_knowledge_chunks()
            print(f"Building FAISS index: encoding {len(knowledge_chunks)} chunks (batch={self.config.EMBEDDING_BATCH_SIZE})...")
            embeddings = self._encode_in_batches(knowledge_chunks, self.config.EMBEDDING_BATCH_SIZE)
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings)
            self.faiss_index.add(embeddings.astype('float32'))
            print(f"✅ FAISS index built with {self.faiss_index.ntotal} vectors")
            fp = self._compute_corpus_fingerprint()
            self._save_index(embeddings.astype('float32'), fp)
            if self.config.USE_HYBRID_RETRIEVAL:
                self.hybrid_retriever = HybridRetriever(
                    faiss_index=self.faiss_index,
                    embedding_model=self.sentence_transformer,
                    document_chunks=self.document_chunks,
                    query_processor=self.query_processor,
                    entity_extractor=self.entity_extractor,
                    cross_encoder=self.cross_encoder,
                    similarity_threshold=self.config.SIMILARITY_THRESHOLD,
                )
                print("✅ Hybrid retriever ready (new index)")
            return True
        except Exception as e:
            print(f"❌ Error creating FAISS index: {e}")
            return False
    
    def initialize_hf_model(self):
        """
        Initialize Hugging Face QA model.
        """
        if not DEPENDENCIES_AVAILABLE:
            print("⚠️ Hugging Face models not available. Using fallback mode.")
            return False
            
        try:
            print("Loading Hugging Face QA model...")
            import torch
            
            # Set device
            if torch.backends.mps.is_available():
                device = 0  # Use MPS
                print("Device set to use mps:0")
            else:
                device = -1  # Use CPU
                print("Device set to use CPU")
            
            self.hf_pipeline = pipeline(
                "question-answering", 
                model=self.config.QA_MODEL,
                token=self.config.HF_TOKEN,
                device=device
            )
            print("✅ Hugging Face QA model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading Hugging Face model: {e}")
            print("Trying fallback initialization...")
            try:
                # Fallback without device specification
                self.hf_pipeline = pipeline(
                    "question-answering", 
                    model=self.config.QA_MODEL,
                    token=self.config.HF_TOKEN
                )
                print("✅ Hugging Face QA model loaded successfully (fallback)!")
                return True
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
                return False
    
    def retrieve(self, query, top_k=None):
        """Hybrid retrieval (vector + lexical + multi-query + NEF) with fallback."""
        if top_k is None:
            top_k = self.config.TOP_K_RETRIEVAL
        if self.hybrid_retriever:
            try:
                results = self.hybrid_retriever.retrieve(query, top_k=top_k)
                # Corrective RAG: if too few or weak scores, attempt expanded second pass
                if not results or (len(results) < max(3, top_k//2)):
                    expanded_queries = self.query_processor.multi_queries(query + " detailed clinical context", max_variants=4)
                    alt = []
                    for q in expanded_queries:
                        alt.extend(self.hybrid_retriever.retrieve(q, top_k=top_k))
                    # merge & dedupe by text hash
                    merged = {}
                    for r in results + alt:
                        key = hash(r['text'][:200])
                        if key not in merged or merged[key]['score'] < r['score']:
                            merged[key] = r
                    results = sorted(merged.values(), key=lambda x: x.get('score',0), reverse=True)[:top_k]
                return results
            except Exception as e:
                print(f"⚠️ Hybrid retrieval failed, fallback to vector-only: {e}")
        # Legacy vector-only retrieval
        if not self.faiss_index or not self.sentence_transformer:
            return self._fallback_retrieve(query, top_k)
        try:
            query_embedding = self.sentence_transformer.encode([query])
            import faiss  # local import
            faiss.normalize_L2(query_embedding)
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
            relevant_chunks = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.document_chunks):
                    chunk = self.document_chunks[idx].copy()
                    chunk['similarity_score'] = float(score)
                    chunk['rank'] = i + 1
                    relevant_chunks.append(chunk)
            relevant_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
            return relevant_chunks
        except Exception as e:
            print(f"❌ Error in fallback vector retrieval: {e}")
            return self._fallback_retrieve(query, top_k)
    
    def _fallback_retrieve(self, query, top_k):
        """
        Fallback retrieval using simple keyword matching.
        """
        query_lower = query.lower()
        relevant_chunks = []
        
        for chunk in self.document_chunks:
            text_lower = chunk['text'].lower()
            # Simple keyword matching
            score = sum(1 for word in query_lower.split() if word in text_lower)
            if score > 0:
                chunk_copy = chunk.copy()
                chunk_copy['similarity_score'] = score / len(query_lower.split())
                relevant_chunks.append(chunk_copy)
        
        # Sort by score and return top_k
        relevant_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
        return relevant_chunks[:top_k]
    
    def _compress_context(self, chunks):
        # Prompt compression: prefer summaries, then highest overlap sentences.
        parts = []
        for c in chunks:
            summ = c.get('summary') or ''
            snippet = summ if 40 < len(summ) < 220 else c['text'][:180]
            parts.append(snippet)
        # Hard cap
        joined = "\n".join(parts)
        if len(joined) > self.config.MAX_CONTEXT_LENGTH:
            joined = joined[: self.config.MAX_CONTEXT_LENGTH]
        return joined

    def generate(self, query, retrieved_chunks):
        """Generate answer using (preferred) OpenRouter LLM or fallback QA model."""
        if not retrieved_chunks:
            return None, []
        # Prefer OpenRouter generative answer
        if self.config.USE_OPENROUTER and self.openrouter_client.is_available:
            try:
                top_use = retrieved_chunks[:5]
                compressed = self._compress_context(top_use)
                system_prompt = (
                    "You are an Ayurvedic assistant. Use ONLY the provided context. "
                    "Cite concepts briefly, avoid hallucination; if unsure, say you are unsure."
                )
                user_prompt = (
                    f"Question: {query}\n\n"
                    f"Context (knowledge snippets):\n{compressed}\n\n"
                    "Provide an evidence-aligned Ayurvedic answer. Structure with: Summary, Key Concepts, Practical Guidance, Safety/Contraindications. If context insufficient, state limitations."
                )
                answer = self.openrouter_client.simple_answer(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=450,
                    temperature=0.3,
                )
                if answer and len(answer) > 20:
                    return answer, retrieved_chunks
            except Exception as e:
                print(f"⚠️ OpenRouter generation failed, fallback to QA: {e}")
        # Fallback QA extractive
        if not self.hf_pipeline:
            return None, retrieved_chunks
        try:
            context_text = " ".join(c['text'] for c in retrieved_chunks[:3])
            if len(context_text) > self.config.MAX_CONTEXT_LENGTH:
                context_text = context_text[: self.config.MAX_CONTEXT_LENGTH]
            if not context_text.strip():
                return None, retrieved_chunks
            result = self.hf_pipeline(
                question=query,
                context=context_text,
                max_answer_len=220,
                handle_impossible_answer=True
            )
            ans = result.get('answer') if isinstance(result, dict) else None
            if isinstance(ans, str) and len(ans.strip()) > 5 and result.get('score', 0) > self.config.CONFIDENCE_THRESHOLD:
                return ans.strip(), retrieved_chunks
            return None, retrieved_chunks
        except Exception as e:
            print(f"❌ Extractive QA generation failed: {e}")
            return None, retrieved_chunks
    
    def rag_pipeline(self, query):
        """
        Complete RAG pipeline: retrieve + generate.
        """
        try:
            # Step 1: Retrieve relevant documents
            retrieved_chunks = self.retrieve(query)
            
            if not retrieved_chunks:
                return None, []
            
            # Step 2: Generate response
            answer, chunks_used = self.generate(query, retrieved_chunks)
            
            return answer, chunks_used
            
        except Exception as e:
            print(f"❌ Error in RAG pipeline: {e}")
            return None, []
    
    def is_ready(self):
        """
        Check if RAG system is fully initialized and ready.
        """
        return len(self.document_chunks) > 0
