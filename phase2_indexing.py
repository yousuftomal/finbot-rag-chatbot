import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
from pathlib import Path

class EmbeddingGenerator:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 384
        print(f"✓ Model loaded (Embedding dimension: {self.embedding_dim})")
    
    def load_chunks(self, filepath='knowledge_base_chunks.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_embeddings(self, chunks, batch_size=32):
        print(f"\nGenerating embeddings for {len(chunks)} chunks...")
        texts = [chunk['text'] for chunk in chunks]
        
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings).astype('float32')
        print(f"✓ Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def save_embeddings(self, embeddings, chunks, output_dir='vector_store'):
        Path(output_dir).mkdir(exist_ok=True)
        
        np.save(f'{output_dir}/embeddings.npy', embeddings)
        with open(f'{output_dir}/chunks_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved embeddings to {output_dir}/embeddings.npy")
        print(f"✓ Saved metadata to {output_dir}/chunks_metadata.json")

class FAISSIndexBuilder:
    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = None
    
    def load_embeddings(self, embeddings_path='vector_store/embeddings.npy'):
        print(f"\nLoading embeddings from {embeddings_path}...")
        embeddings = np.load(embeddings_path)
        print(f"✓ Loaded embeddings shape: {embeddings.shape}")
        return embeddings
    
    def load_metadata(self, metadata_path='vector_store/chunks_metadata.json'):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def build_index(self, embeddings, index_type='flat'):
        print(f"\nBuilding FAISS index (type: {index_type})...")
        
        if index_type == 'flat':
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif index_type == 'ivf':
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            nlist = min(100, embeddings.shape[0] // 10)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            self.index.train(embeddings)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.index.add(embeddings)
        print(f"✓ Index built with {self.index.ntotal} vectors")
    
    def save_index(self, output_path='vector_store/faiss_index.bin'):
        faiss.write_index(self.index, output_path)
        print(f"✓ Saved FAISS index to {output_path}")
    
    def get_index_stats(self):
        print("\n" + "="*50)
        print("=== FAISS Index Statistics ===")
        print("="*50)
        print(f"Total vectors: {self.index.ntotal}")
        print(f"Index dimension: {self.embedding_dim}")
        print(f"Index type: {type(self.index).__name__}")
        print(f"Is trained: {self.index.is_trained}")
        print("="*50)

class RetrievalEngine:
    def __init__(self, 
                 index_path='vector_store/faiss_index.bin',
                 metadata_path='vector_store/chunks_metadata.json',
                 model_name='sentence-transformers/all-MiniLM-L6-v2'):
        print("Initializing Retrieval Engine...")
        
        self.index = faiss.read_index(index_path)
        print(f"✓ Loaded FAISS index with {self.index.ntotal} vectors")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"✓ Loaded metadata for {len(self.metadata)} chunks")
        
        self.model = SentenceTransformer(model_name)
        print(f"✓ Loaded embedding model")
    
    def retrieve(self, query, top_k=4):
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def test_retrieval(self, test_queries):
        print("\n" + "="*50)
        print("=== Testing Retrieval Engine ===")
        print("="*50)
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = self.retrieve(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                print(f"\n[{i}] Score: {result['similarity_score']:.4f}")
                print(f"Source: {result['source']}")
                print(f"Text: {result['text'][:150]}...")
            print("-" * 50)

def main():
    print("="*50)
    print("Phase 2: Indexing & Retrieval Engine")
    print("="*50)
    
    print("\nStep 1: Generate Embeddings")
    print("-" * 50)
    generator = EmbeddingGenerator()
    chunks = generator.load_chunks('knowledge_base_chunks.json')
    embeddings = generator.generate_embeddings(chunks, batch_size=32)
    generator.save_embeddings(embeddings, chunks)
    
    print("\n" + "="*50)
    print("\nStep 2: Build FAISS Index")
    print("-" * 50)
    builder = FAISSIndexBuilder(embedding_dim=384)
    embeddings = builder.load_embeddings()
    metadata = builder.load_metadata()
    builder.build_index(embeddings, index_type='flat')
    builder.save_index()
    builder.get_index_stats()
    
    print("\n" + "="*50)
    print("\nStep 3: Test Retrieval Engine")
    print("-" * 50)
    retrieval_engine = RetrievalEngine()
    
    test_queries = [
        "What is a 401k retirement plan?",
        "How does credit score work?",
        "What is the difference between stocks and bonds?",
        "How to build an emergency fund?"
    ]
    
    retrieval_engine.test_retrieval(test_queries)
    
    print("\n✓ Phase 2 Complete!")
    print("\nOutput Files:")
    print("  • vector_store/embeddings.npy")
    print("  • vector_store/chunks_metadata.json")
    print("  • vector_store/faiss_index.bin")
    print("\nRetrieval engine is ready for Phase 3!")

if __name__ == "__main__":
    main()