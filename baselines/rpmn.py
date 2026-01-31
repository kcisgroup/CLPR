# rpmn.py
"""
RPMN (Re-finding Personalized Memory Network)
- Uses SentenceTransformer for query encoding.
- Adapted for MedCorpus multi-turn conversations.
- Fixed AttributeError for integer query_ids.
"""
import json
import logging
import os
import sys
from typing import List, Dict, Any
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import argparse

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RPMN')

# --- PyTorch and FAISS Check ---
try:
    import torch
    import faiss
except ImportError as e:
    logger.error(f"Missing required library: {e}. Please install torch and faiss-cpu or faiss-gpu.")
    sys.exit(1)

class RPMNBaseline:
    """RPMN基线方法实现"""

    def __init__(self, model_path=None, device=None, batch_size=16):
        self.logger = logger
        self.encoder_model_path = model_path
        if not self.encoder_model_path:
             raise ValueError("Encoder model_path (e.g., all-MiniLM-L6-v2) is required for RPMN.")
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.logger.info(f"RPMN Initializing with encoder: {self.encoder_model_path}, device: {self.device}, batch_size: {self.batch_size}")

        self.short_term_memory: Dict[str, List[Dict[str, np.ndarray]]] = {} # topic_id -> list of {"embedding": np.ndarray}
        self.medium_term_memory: Dict[str, List[Dict[str, np.ndarray]]] = {}# topic_id -> list of {"embedding": np.ndarray}
        self.long_term_memory: Dict[str, Dict[str, Any]] = {} # topic_id -> {"count": int, "avg_embedding": np.ndarray}

        self.encoder_st_model = None
        self.embedding_dim = 0
        self._setup_encoder()

    def _setup_encoder(self):
        """Loads the SentenceTransformer encoder model."""
        try:
            self.logger.info(f"RPMN: Loading SentenceTransformer encoder: {self.encoder_model_path}")
            use_trust_remote_code = "gte" in self.encoder_model_path.lower() or "modelscope" in self.encoder_model_path.lower()
            self.encoder_st_model = SentenceTransformer(
                self.encoder_model_path,
                device=self.device,
                trust_remote_code=use_trust_remote_code
            )
            self.embedding_dim = self.encoder_st_model.get_sentence_embedding_dimension()
            self.logger.info(f"RPMN: SentenceTransformer encoder loaded to {self.device}. Embedding dim: {self.embedding_dim}")
        except Exception as e:
            self.logger.error(f"RPMN: Failed to load SentenceTransformer encoder: {e}", exc_info=True)
            raise

    def _encode_text(self, texts: List[str]) -> np.ndarray:
        """Encodes a list of texts to embeddings using SentenceTransformer and normalizes them."""
        if not texts: return np.array([], dtype=np.float32)
        if not self.encoder_st_model:
            self.logger.error("RPMN: SentenceTransformer encoder not initialized for encoding.")
            return np.zeros((len(texts), self.embedding_dim if self.embedding_dim > 0 else 384), dtype=np.float32)
        try:
            embeddings = self.encoder_st_model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                device=self.device,
                show_progress_bar=False
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            self.logger.error(f"RPMN: Error encoding texts: {e}", exc_info=True)
            return np.zeros((len(texts), self.embedding_dim if self.embedding_dim > 0 else 384), dtype=np.float32)


    def _update_memories(self, query_text: str, history_query_texts: List[str], topic_id: str):
        """更新特定topic_id的记忆网络"""
        try:
            query_embedding_array = self._encode_text([query_text])
            if query_embedding_array.size == 0 or query_embedding_array.ndim == 0:
                self.logger.warning(f"RPMN: Failed to encode query '{query_text}' for memory update in topic '{topic_id}'. Skipping.")
                return
            current_query_embedding = query_embedding_array[0]

            # Short-term memory
            if topic_id not in self.short_term_memory: self.short_term_memory[topic_id] = []
            self.short_term_memory[topic_id].append({"embedding": current_query_embedding})
            self.short_term_memory[topic_id] = self.short_term_memory[topic_id][-5:] # Keep last 5

            # Medium-term memory
            if topic_id not in self.medium_term_memory: self.medium_term_memory[topic_id] = []
            # Add history embeddings (if any) and current query embedding
            all_medium_term_texts = history_query_texts + [query_text]
            if all_medium_term_texts:
                medium_embeddings = self._encode_text(all_medium_term_texts)
                if medium_embeddings.size > 0:
                    # Replace medium term memory with embeddings from the current relevant window
                    self.medium_term_memory[topic_id] = [{"embedding": emb} for emb in medium_embeddings]
            self.medium_term_memory[topic_id] = self.medium_term_memory[topic_id][-10:] # Keep last 10 relevant turns

            # Long-term memory
            if topic_id not in self.long_term_memory:
                self.long_term_memory[topic_id] = {"count": 0, "avg_embedding": np.zeros(self.embedding_dim, dtype=np.float32)}

            memory_lt = self.long_term_memory[topic_id]
            memory_lt["count"] += 1
            memory_lt["avg_embedding"] = memory_lt["avg_embedding"] + (current_query_embedding - memory_lt["avg_embedding"]) / memory_lt["count"]
        except Exception as e:
            self.logger.error(f"RPMN: Error updating memories for topic {topic_id}, query '{query_text}': {e}", exc_info=True)

    def _get_personalized_vector(self, query_text: str, history_query_texts: List[str], topic_id: str) -> np.ndarray:
        """获取特定topic_id的个性化查询向量"""
        self._update_memories(query_text, history_query_texts, topic_id) # Ensure memories are current for this query

        current_query_embedding_array = self._encode_text([query_text])
        if current_query_embedding_array.size == 0 or current_query_embedding_array.ndim == 0:
             self.logger.warning(f"RPMN: Failed to encode query '{query_text}' for personalized vector in topic '{topic_id}'. Returning zero vector.")
             return np.zeros(self.embedding_dim, dtype=np.float32)
        current_query_embedding = current_query_embedding_array[0]

        short_term_vec = current_query_embedding
        if topic_id in self.short_term_memory and self.short_term_memory[topic_id]:
            short_term_vec = np.mean([m["embedding"] for m in self.short_term_memory[topic_id]], axis=0)


        medium_term_vec = np.zeros_like(current_query_embedding)
        if topic_id in self.medium_term_memory and self.medium_term_memory[topic_id]:
             session_embeddings = [m["embedding"] for m in self.medium_term_memory[topic_id]]
             if session_embeddings:
                 medium_term_vec = np.mean(session_embeddings, axis=0)

        long_term_vec = np.zeros_like(current_query_embedding)
        if topic_id in self.long_term_memory and self.long_term_memory[topic_id]["count"] > 0 :
             long_term_vec = self.long_term_memory[topic_id]["avg_embedding"]

        w_curr = 0.6; w_short = 0.2; w_medium = 0.15; w_long = 0.05
        personalized_vector = (w_curr * current_query_embedding +
                               w_short * short_term_vec +
                               w_medium * medium_term_vec +
                               w_long * long_term_vec)

        norm = np.linalg.norm(personalized_vector)
        if norm > 1e-12:
            personalized_vector = personalized_vector / norm
        else:
            self.logger.warning(f"RPMN: Norm of personalized vector is zero for query '{query_text}', topic '{topic_id}'. Using current query embedding.")
            return current_query_embedding

        return personalized_vector.astype(np.float32)

    def load_corpus_and_embeddings(self, corpus_path, corpus_embeddings_path) -> tuple[Dict[str, Any], List[str], np.ndarray]:
        self.logger.info(f"RPMN: Loading corpus from {corpus_path}")
        corpus: Dict[str, Any] = {}
        doc_ids_list: List[str] = []
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        doc_id = str(data.get('text_id', '')) # Ensure text_id is string
                        if doc_id:
                            corpus[doc_id] = data
                            doc_ids_list.append(doc_id)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Skipping invalid JSON line in corpus: {line[:100]}...")
        except Exception as e:
            self.logger.error(f"RPMN: Error loading corpus: {e}", exc_info=True); raise

        self.logger.info(f"RPMN: Loading corpus embeddings from {corpus_embeddings_path}")
        try:
            corpus_embeddings_arr = np.load(corpus_embeddings_path).astype(np.float32)
            if corpus_embeddings_arr.ndim != 2 or corpus_embeddings_arr.shape[0] != len(doc_ids_list):
                 raise ValueError(f"Embeddings shape mismatch: Expected ({len(doc_ids_list)}, dim), Got {corpus_embeddings_arr.shape}")
        except Exception as e:
            self.logger.error(f"RPMN: Error loading/validating embeddings: {e}", exc_info=True); raise
        self.logger.info(f"RPMN: Loaded {len(corpus)} documents and embeddings shape {corpus_embeddings_arr.shape}")
        return corpus, doc_ids_list, corpus_embeddings_arr

    def search(self, query_text: str, history_query_texts: List[str], topic_id: str,
               index: faiss.Index, doc_ids_list: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        if not index or not doc_ids_list:
            self.logger.error("RPMN: FAISS index or doc_ids not provided for search.")
            return []
        try:
            query_vector_personalized = self._get_personalized_vector(query_text, history_query_texts, topic_id)
            if query_vector_personalized.size == 0: return []
            query_vector_np = query_vector_personalized.reshape(1, -1)

            scores, indices = index.search(query_vector_np, top_k)
            results: List[Dict[str, Any]] = []
            if scores.size > 0 and indices.size > 0:
                 for score_val, idx_val in zip(scores[0], indices[0]):
                     if idx_val != -1 and idx_val < len(doc_ids_list): # Check for valid index
                         results.append({"text_id": doc_ids_list[idx_val], "score": float(score_val)})
            return results
        except Exception as e:
            self.logger.error(f"RPMN: Error during FAISS search for topic {topic_id}, query '{query_text}': {e}", exc_info=True)
            return []

    def process_dataset(self, input_file, output_file, corpus_path, corpus_embeddings_path):
        if not all(os.path.exists(p) for p in [input_file, corpus_path, corpus_embeddings_path]):
            self.logger.error("RPMN: One or more input files not found. Exiting.")
            return
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        self.logger.info(f"RPMN: Processing queries from {input_file}")

        try:
            corpus_dict, doc_ids_list_corpus, corpus_embeddings_arr = self.load_corpus_and_embeddings(corpus_path, corpus_embeddings_path)
            if self.embedding_dim == 0 and corpus_embeddings_arr.ndim == 2:
                self.embedding_dim = corpus_embeddings_arr.shape[1]

            index = faiss.IndexFlatIP(corpus_embeddings_arr.shape[1])
            index.add(corpus_embeddings_arr)
            self.logger.info(f"RPMN: FAISS index created with {index.ntotal} vectors.")

            queries_data_all = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try: queries_data_all.append(json.loads(line.strip()))
                    except json.JSONDecodeError: self.logger.warning(f"Skipping invalid JSON line in input: {line[:100]}...")

            final_results_to_save: List[Dict[str, Any]] = []
            processed_count = 0
            grouped_queries_by_topic: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for item_query_data in queries_data_all:
                 # Ensure query_id is treated as string for consistent topic_id derivation
                 query_id_str_item = str(item_query_data.get("query_id", ""))
                 topic_id_parts = query_id_str_item.split('_')
                 # Robust topic_id derivation
                 current_topic_id = "_".join(topic_id_parts[:-1]) if len(topic_id_parts) > 1 and topic_id_parts[-1].isdigit() else query_id_str_item
                 grouped_queries_by_topic[current_topic_id].append(item_query_data)

            for topic_id_key, items_in_current_topic in tqdm(grouped_queries_by_topic.items(), desc="RPMN: Processing topics/sessions"):
                 def get_turn_id_from_item(q_item_data: Dict[str, Any]) -> int:
                     # Ensure qid_item is a string before splitting
                     qid_item = str(q_item_data.get("query_id", "0")) # Ensure string
                     parts_item = qid_item.split('_')
                     return int(parts_item[-1]) if len(parts_item) > 1 and parts_item[-1].isdigit() else 0
                 items_in_current_topic.sort(key=get_turn_id_from_item)

                 # Initialize/reset memories for the new topic at the beginning of processing its turns
                 self.short_term_memory[topic_id_key] = []
                 self.medium_term_memory[topic_id_key] = []
                 if self.embedding_dim > 0: # Ensure embedding_dim is known
                    self.long_term_memory[topic_id_key] = {"count": 0, "avg_embedding": np.zeros(self.embedding_dim, dtype=np.float32)}
                 else: # Fallback if embedding_dim isn't set (should not happen if encoder setup is correct)
                    self.logger.warning(f"RPMN: embedding_dim is 0 for topic {topic_id_key}. LTM might not initialize correctly.")
                    # Attempt to initialize with a common default if necessary, or handle error
                    default_dim_for_init = 384 # Example default
                    self.long_term_memory[topic_id_key] = {"count": 0, "avg_embedding": np.zeros(default_dim_for_init, dtype=np.float32)}


                 current_topic_history_texts: List[str] = []

                 for item_data_turn in items_in_current_topic:
                     current_query_id_str = str(item_data_turn.get("query_id", "")) # Ensure string
                     current_query_text = item_data_turn.get("query", "")
                     if not current_query_id_str or not current_query_text:
                         self.logger.warning(f"RPMN: Skipping item with missing qid or query in topic {topic_id_key}")
                         continue

                     search_results_list = self.search(
                         query_text=current_query_text,
                         history_query_texts=list(current_topic_history_texts),
                         topic_id=topic_id_key,
                         index=index,
                         doc_ids_list=doc_ids_list_corpus,
                         top_k=10
                     )
                     # After search, _get_personalized_vector (called by search) would have updated memories
                     # including the current_query_text. Add it to explicit history for next turn.
                     current_topic_history_texts.append(current_query_text)
                     current_topic_history_texts = current_topic_history_texts[-5:] # Keep relevant window for history context


                     full_results_for_query: List[Dict[str, Any]] = []
                     for res_item in search_results_list:
                         doc_id_res_item = res_item["text_id"]
                         if doc_id_res_item in corpus_dict:
                             doc_data_item = corpus_dict[doc_id_res_item]
                             full_results_for_query.append({
                                 "text_id": doc_id_res_item,
                                 "title": doc_data_item.get("title", ""),
                                 "text": doc_data_item.get("text", ""),
                                 "score": res_item["score"]
                             })
                         else:
                             self.logger.warning(f"RPMN: Doc ID {doc_id_res_item} not found in corpus for query {current_query_id_str}.")
                     final_results_to_save.append({
                         "query_id": current_query_id_str,
                         "query": current_query_text, # Store original query for this baseline
                         "results": full_results_for_query
                     })
                     processed_count += 1

            with open(output_file, 'w', encoding='utf-8') as f_out:
                for result_entry in final_results_to_save:
                    f_out.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
            self.logger.info(f"RPMN: Processed {processed_count} queries. Results saved to {output_file}")

        except Exception as e:
            self.logger.error(f"RPMN: Error processing dataset: {e}", exc_info=True); raise

def main():
    parser = argparse.ArgumentParser(description='RPMN Baseline - Re-finding Personalized Memory Network')
    parser.add_argument('--input_file', type=str, required=True, help='Input queries file path (JSONL format)')
    parser.add_argument('--output_file', type=str, required=True, help='Output results file path (JSONL)')
    parser.add_argument('--corpus_path', type=str, required=True, help='Corpus file path (JSONL format)')
    parser.add_argument('--corpus_embeddings_path', type=str, required=True, help='Corpus embeddings file path (.npy)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the SentenceTransformer encoder model (e.g., all-MiniLM-L6-v2)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for encoding')
    parser.add_argument('--device', type=str, default=None, help='Device for inference (e.g., cuda:0, cpu)')

    args = parser.parse_args()

    if args.device is None:
         args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
         logger.info(f"RPMN: Device not specified, using: {args.device}")
    elif not (args.device == "cpu" or (args.device.startswith("cuda:") and args.device.split(':')[1].isdigit())):
         logger.error(f"RPMN: Invalid device format: '{args.device}'. Use 'cpu' or 'cuda:N'. Exiting.")
         sys.exit(1)

    try:
        rpmn_instance = RPMNBaseline(model_path=args.model_path, device=args.device, batch_size=args.batch_size)
        rpmn_instance.process_dataset(args.input_file, args.output_file, args.corpus_path, args.corpus_embeddings_path)
    except Exception as e:
         logger.error(f"RPMN execution failed: {e}", exc_info=True)
         sys.exit(1)

if __name__ == '__main__':
    main()
