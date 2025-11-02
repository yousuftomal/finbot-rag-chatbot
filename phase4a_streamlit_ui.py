import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv
from typing import List, Dict
import time

load_dotenv()

st.set_page_config(
    page_title="FinBot - Financial Literacy Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RAGPipeline:
    def __init__(self,
                 index_path='vector_store/faiss_index.bin',
                 metadata_path='vector_store/chunks_metadata.json',
                 model_name='sentence-transformers/all-MiniLM-L6-v2',
                 groq_api_key=None,
                 similarity_threshold=0.5):
        
        self.similarity_threshold = similarity_threshold
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.embedding_model = SentenceTransformer(model_name)
        
        if groq_api_key:
            self.groq_client = Groq(api_key=groq_api_key)
        else:
            self.groq_client = None
    
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
                'answer': "Error: Groq API client not initialized. Please check your API key.",
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

@st.cache_resource
def load_rag_pipeline():
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        st.error("⚠️ GROQ_API_KEY not found! Please set it in your .env file.")
        st.stop()
    return RAGPipeline(groq_api_key=groq_api_key, similarity_threshold=0.5)

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = True

def display_sources(retrieved_chunks):
    if not retrieved_chunks:
        st.info("ℹ️ No sources retrieved (query out of knowledge base scope)")
        return
    
    st.markdown("### 📚 Retrieved Sources")
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        with st.expander(f"Source {i}: {chunk['source']} (Score: {chunk['similarity_score']:.4f})"):
            st.markdown(f"**Text:** {chunk['text'][:300]}...")
            st.caption(f"URL: {chunk['url']}")

def main():
    init_session_state()
    rag = load_rag_pipeline()
    
    with st.sidebar:
        st.title("💰 FinBot")
        st.markdown("### Financial Literacy Assistant")
        st.markdown("---")
        
        st.markdown("#### 📊 Knowledge Base Stats")
        st.metric("Total Chunks", "1,101")
        st.metric("Documents", "47")
        st.metric("Sources", "3 (CFPB, Investopedia, Investor.gov)")
        
        st.markdown("---")
        
        st.markdown("#### 💡 Example Questions")
        example_questions = [
            "What is a 401k retirement plan?",
            "How does credit score work?",
            "What is the difference between stocks and bonds?",
            "How to build an emergency fund?",
            "What is a Roth IRA?"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"example_{question}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
        
        st.markdown("---")
        
        st.session_state.show_sources = st.checkbox("Show Sources", value=True)
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("#### ℹ️ About")
        st.markdown("""
        **FinBot** uses Retrieval-Augmented Generation (RAG) to provide accurate financial education.
        
        **Features:**
        - ✅ Only answers from verified sources
        - ✅ Transparent source citation
        - ✅ Refuses questions outside knowledge base
        - ✅ No personalized financial advice
        """)
    
    st.title("💰 FinBot - Your Financial Literacy Assistant")
    st.markdown("Ask me anything about personal finance, investing, credit, retirement planning, and more!")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sources" in message and st.session_state.show_sources:
                with st.expander("📚 View Sources"):
                    display_sources(message["sources"])
    
    if prompt := st.chat_input("Ask a financial question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag.generate_response(prompt)
            
            st.markdown(response['answer'])
            
            if st.session_state.show_sources:
                with st.expander("📚 View Sources"):
                    display_sources(response['retrieved_chunks'])
            
            st.caption(f"⏱️ Response time: {response['total_time']:.2f}s (Retrieval: {response['retrieval_time']:.3f}s | Generation: {response['generation_time']:.3f}s)")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response['answer'],
            "sources": response['retrieved_chunks']
        })

if __name__ == "__main__":
    main()
