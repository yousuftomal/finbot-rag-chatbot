import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from pathlib import Path
from typing import List, Dict
import time

from dotenv import load_dotenv
load_dotenv()  # This loads the .env file

class RAGPipeline:
    def __init__(self,
                 index_path='vector_store/faiss_index.bin',
                 metadata_path='vector_store/chunks_metadata.json',
                 model_name='sentence-transformers/all-MiniLM-L6-v2',
                 groq_api_key= None,
                 similarity_threshold=0.5):
        
        print("="*50)
        print("Initializing RAG Pipeline...")
        print("="*50)
        
        self.similarity_threshold = similarity_threshold
        
        print("\n[1/3] Loading FAISS index...")
        self.index = faiss.read_index(index_path)
        print(f"✓ Loaded index with {self.index.ntotal} vectors")
        
        print("\n[2/3] Loading metadata...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print(f"✓ Loaded {len(self.metadata)} chunk metadata")
        
        print("\n[3/3] Loading embedding model...")
        self.embedding_model = SentenceTransformer(model_name)
        print(f"✓ Embedding model ready")
        
        if groq_api_key:
            self.groq_client = Groq(api_key=groq_api_key)
            print(f"✓ Groq API client initialized")
        else:
            print("⚠ No Groq API key provided. Set GROQ_API_KEY environment variable.")
            self.groq_client = None
        
        print("\n" + "="*50)
        print("RAG Pipeline Ready!")
        print("="*50)
    
    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                if score >= self.similarity_threshold:
                    result = self.metadata[idx].copy()
                    result['similarity_score'] = float(score)
                    results.append(result)
        
        return results
    
    def format_context(self, retrieved_chunks: List[Dict]) -> str:
        if not retrieved_chunks:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[Source {i} - {chunk['source']}]")
            context_parts.append(chunk['text'])
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def build_prompt(self, query: str, context: str) -> str:
        prompt_template = """You are FinBot, a helpful and trustworthy financial literacy assistant.

Your STRICT instructions:
1. Answer the user's question ONLY using the information provided in the CONTEXT below
2. Do NOT use any outside knowledge or make assumptions
3. If the context does not contain enough information to answer the question, respond: "I don't have enough information in my knowledge base to answer that question accurately. Please ask about topics like retirement planning, credit scores, investing, mortgages, or savings."
4. Keep answers clear, educational, and factual
5. Never provide personalized financial advice (e.g., "You should buy X stock")
6. Cite sources by mentioning them (e.g., "According to the CFPB...")

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        
        return prompt_template.format(context=context, question=query)
    
    def generate_response(self, query: str, model: str = "llama-3.1-8b-instant") -> Dict:
        start_time = time.time()
        
        retrieved_chunks = self.retrieve(query, top_k=4)
        retrieval_time = time.time() - start_time
        
        if not retrieved_chunks:
            return {
                'query': query,
                'answer': "I don't have enough information in my knowledge base to answer that question accurately. Please ask about topics like retirement planning (401k, IRA), credit scores, investing (stocks, bonds, ETFs), mortgages, insurance, taxes, or savings.",
                'retrieved_chunks': [],
                'retrieval_time': retrieval_time,
                'generation_time': 0,
                'total_time': retrieval_time
            }
        
        context = self.format_context(retrieved_chunks)
        prompt = self.build_prompt(query, context)
        
        if not self.groq_client:
            return {
                'query': query,
                'answer': "Error: Groq API client not initialized. Please provide API key.",
                'retrieved_chunks': retrieved_chunks,
                'retrieval_time': retrieval_time,
                'generation_time': 0,
                'total_time': retrieval_time
            }
        
        gen_start = time.time()
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are FinBot, a factual financial education assistant. Only use provided context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=model,
                temperature=0.3,
                max_tokens=512,
                top_p=0.9,
                stream=False
            )
            
            answer = chat_completion.choices[0].message.content
            generation_time = time.time() - gen_start
            
            return {
                'query': query,
                'answer': answer,
                'retrieved_chunks': retrieved_chunks,
                'retrieval_time': retrieval_time,
                'generation_time': generation_time,
                'total_time': retrieval_time + generation_time,
                'model': model
            }
        
        except Exception as e:
            return {
                'query': query,
                'answer': f"Error generating response: {str(e)}",
                'retrieved_chunks': retrieved_chunks,
                'retrieval_time': retrieval_time,
                'generation_time': 0,
                'total_time': retrieval_time
            }
    
    def display_response(self, response: Dict):
        print("\n" + "="*70)
        print(f"Query: {response['query']}")
        print("="*70)
        
        print("\n📚 Retrieved Context:")
        print("-"*70)
        if response['retrieved_chunks']:
            for i, chunk in enumerate(response['retrieved_chunks'], 1):
                print(f"\n[{i}] Score: {chunk['similarity_score']:.4f} | Source: {chunk['source']}")
                print(f"Text: {chunk['text'][:200]}...")
        else:
            print("No relevant chunks found (below similarity threshold)")
        
        print("\n" + "-"*70)
        print("\n💡 FinBot Answer:")
        print("-"*70)
        print(response['answer'])
        
        print("\n" + "-"*70)
        print(f"⏱️  Retrieval: {response['retrieval_time']:.3f}s | "
              f"Generation: {response['generation_time']:.3f}s | "
              f"Total: {response['total_time']:.3f}s")
        print("="*70)

def test_rag_pipeline():
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    if not groq_api_key:
        print("\n⚠️  WARNING: GROQ_API_KEY not found in environment variables!")
        print("\nTo set up Groq API:")
        print("1. Visit: https://console.groq.com/")
        print("2. Sign up (free)")
        print("3. Create API key")
        print("4. Set environment variable:")
        print("   - Windows: set GROQ_API_KEY=your_key_here")
        print("   - Linux/Mac: export GROQ_API_KEY=your_key_here")
        print("   - Or create .env file with: GROQ_API_KEY=your_key_here\n")
        return
    
    rag = RAGPipeline(groq_api_key=groq_api_key, similarity_threshold=0.5)
    
    test_queries = [
        "What is a 401k retirement plan?",
        "How does credit score work?",
        "What is the difference between stocks and bonds?",
        "Should I invest in cryptocurrency?",
        "How much should I save for retirement?"
    ]
    
    print("\n" + "="*70)
    print("Testing RAG Pipeline with Financial Queries")
    print("="*70)
    
    for query in test_queries:
        response = rag.generate_response(query)
        rag.display_response(response)
        print("\n")
        time.sleep(1)

def interactive_mode():
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    if not groq_api_key:
        print("\n❌ Error: GROQ_API_KEY not found!")
        print("Please set your Groq API key as an environment variable.")
        return
    
    rag = RAGPipeline(groq_api_key=groq_api_key, similarity_threshold=0.5)
    
    print("\n" + "="*70)
    print("🤖 FinBot Interactive Mode")
    print("="*70)
    print("Ask me anything about personal finance!")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            query = input("\n💬 You: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! Stay financially literate! 💰")
                break
            
            if not query:
                continue
            
            response = rag.generate_response(query)
            rag.display_response(response)
        
        except KeyboardInterrupt:
            print("\n\nGoodbye! Stay financially literate! 💰")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

def main():
    print("="*70)
    print("Phase 3: RAG Pipeline (Generation Engine)")
    print("="*70)
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_mode()
    else:
        test_rag_pipeline()
        print("\n✓ Phase 3 Complete!")
        print("\nTo use interactive mode, run:")
        print("  python phase3_rag_pipeline.py --interactive")

if __name__ == "__main__":
    main()