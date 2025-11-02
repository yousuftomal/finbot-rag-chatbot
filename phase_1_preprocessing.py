import requests
from bs4 import BeautifulSoup
import json
import time
import re
from pathlib import Path

class FinancialKnowledgeBaseScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        self.documents = []
        
    def scrape_cfpb(self):
        urls = [
            'https://www.consumerfinance.gov/consumer-tools/educator-tools/your-money-your-goals/',
            'https://www.consumerfinance.gov/consumer-tools/money-as-you-grow/',
            'https://www.consumerfinance.gov/consumer-tools/credit-reports-and-scores/',
            'https://www.consumerfinance.gov/consumer-tools/mortgages/',
            'https://www.consumerfinance.gov/consumer-tools/student-loans/',
            'https://www.consumerfinance.gov/consumer-tools/debt-collection/',
            'https://www.consumerfinance.gov/consumer-tools/retirement/',
            'https://www.consumerfinance.gov/consumer-tools/bank-accounts/',
            'https://www.consumerfinance.gov/consumer-tools/credit-cards/',
            'https://www.consumerfinance.gov/consumer-tools/auto-loans/',
            'https://www.consumerfinance.gov/ask-cfpb/what-is-a-debt-to-income-ratio-en-1791/',
            'https://www.consumerfinance.gov/ask-cfpb/how-do-i-get-a-copy-of-my-credit-reports-en-5/',
            'https://www.consumerfinance.gov/ask-cfpb/what-is-a-credit-score-en-315/',
            'https://www.consumerfinance.gov/owning-a-home/process/',
            'https://www.consumerfinance.gov/paying-for-college/'
        ]
        
        for url in urls:
            try:
                response = requests.get(url, headers=self.headers, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
                        tag.decompose()
                    
                    content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
                    if content:
                        text = ' '.join([p.get_text() for p in content.find_all(['p', 'h1', 'h2', 'h3', 'li'])])
                        text = re.sub(r'\s+', ' ', text).strip()
                        
                        if len(text) > 300:
                            self.documents.append({
                                'source': 'CFPB',
                                'url': url,
                                'text': text
                            })
                            print(f"✓ Scraped: {url.split('/')[-2]}")
                
                time.sleep(1.5)
            except Exception as e:
                print(f"✗ Failed {url}: {str(e)}")
    
    def scrape_investopedia_comprehensive(self):
        topics = [
            ('r', 'retirement'), ('r', 'roth-ira'), ('i', 'ira'), ('t', '401k'), ('t', '403b'),
            ('i', 'investing'), ('c', 'credit'), ('m', 'mortgage'), ('i', 'insurance'),
            ('t', 'taxes'), ('s', 'savings'), ('b', 'budgeting'), ('s', 'stocks'), ('b', 'bonds'),
            ('m', 'mutual-funds'), ('e', 'etf'), ('c', 'credit-score'), ('c', 'compound-interest'),
            ('a', 'asset-allocation'), ('d', 'diversification'), ('r', 'risk-tolerance'),
            ('e', 'emergency-fund'), ('d', 'debt-management'), ('s', 'student-loans'),
            ('a', 'apr'), ('c', 'credit-card'), ('r', 'refinance'), ('h', 'home-equity'),
            ('i', 'index-fund'), ('d', 'dividend'), ('c', 'capital-gains'), ('i', 'inflation'),
            ('n', 'net-worth'), ('a', 'amortization'), ('e', 'escrow'), ('d', 'down-payment'),
            ('c', 'closing-costs'), ('p', 'preapproval'), ('f', 'fico-score'),
            ('d', 'debt-to-income-ratio'), ('a', 'annual-percentage-rate'), ('v', 'vesting'),
            ('b', 'beneficiary'), ('e', 'estate-planning'), ('w', 'will'), ('t', 'trust'),
            ('l', 'life-insurance'), ('h', 'health-savings-account'), ('f', 'flexible-spending-account'),
            ('m', 'medicare'), ('s', 'social-security'), ('p', 'pension'), ('a', 'annuity'),
            ('b', 'bankruptcy'), ('f', 'foreclosure'), ('c', 'consolidation'),
            ('r', 'rebalancing'), ('d', 'dollar-cost-averaging'), ('b', 'blue-chip'),
            ('g', 'growth-stock'), ('v', 'value-stock'), ('m', 'market-capitalization'),
            ('p', 'portfolio'), ('b', 'bear-market'), ('b', 'bull-market'), ('r', 'recession')
        ]
        
        for letter, topic in topics:
            url = f'https://www.investopedia.com/terms/{letter}/{topic}.asp'
            try:
                response = requests.get(url, headers=self.headers, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'figure', 'ad']):
                        tag.decompose()
                    
                    article = soup.find('article') or soup.find('div', {'id': 'article-body_1-0'})
                    if article:
                        text = ' '.join([p.get_text() for p in article.find_all(['p', 'h2', 'h3', 'li'])])
                        text = re.sub(r'\s+', ' ', text).strip()
                        
                        if len(text) > 300:
                            self.documents.append({
                                'source': 'Investopedia',
                                'url': url,
                                'text': text
                            })
                            print(f"✓ Scraped: {topic}")
                
                time.sleep(1.5)
            except Exception as e:
                print(f"✗ Failed {topic}: {str(e)}")
    
    def scrape_investor_gov_fixed(self):
        urls = [
            'https://www.investor.gov/introduction-investing/investing-basics',
            'https://www.investor.gov/introduction-investing/investing-basics/save-and-invest',
            'https://www.investor.gov/introduction-investing/investing-basics/investment-products',
            'https://www.investor.gov/introduction-investing/investing-basics/how-stock-markets-work',
            'https://www.investor.gov/introduction-investing/investing-basics/investment-products/stocks',
            'https://www.investor.gov/introduction-investing/investing-basics/investment-products/bonds',
            'https://www.investor.gov/introduction-investing/investing-basics/investment-products/mutual-funds',
            'https://www.investor.gov/introduction-investing/investing-basics/investment-products/etfs',
            'https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins/how-fees',
            'https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins/what-are',
            'https://www.investor.gov/additional-resources/retirement-toolkit'
        ]
        
        for url in urls:
            try:
                response = requests.get(url, headers=self.headers, timeout=15, allow_redirects=True)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'form']):
                        tag.decompose()
                    
                    main_content = (soup.find('div', class_='layout-content') or 
                                  soup.find('main') or 
                                  soup.find('article') or
                                  soup.find('div', {'role': 'main'}))
                    
                    if main_content:
                        paragraphs = main_content.find_all(['p', 'li', 'h2', 'h3'])
                        text = ' '.join([p.get_text(strip=True) for p in paragraphs])
                        text = re.sub(r'\s+', ' ', text).strip()
                        
                        if len(text) > 300:
                            self.documents.append({
                                'source': 'Investor.gov',
                                'url': url,
                                'text': text
                            })
                            print(f"✓ Scraped: {url.split('/')[-1]}")
                
                time.sleep(2)
            except Exception as e:
                print(f"✗ Failed {url}: {str(e)}")
    
    def scrape_nerdwallet(self):
        topics = [
            'what-is-a-good-credit-score',
            'how-to-build-credit',
            'how-to-save-money',
            'what-is-a-roth-ira',
            'what-is-a-401k',
            'emergency-fund',
            'budgeting-101',
            'debt-avalanche-vs-debt-snowball',
            'investing-for-beginners',
            'index-funds',
            'high-yield-savings-accounts',
            'cd-rates',
            'mortgage-rates',
            'refinance-calculator',
            'student-loan-refinance'
        ]
        
        for topic in topics:
            url = f'https://www.nerdwallet.com/article/finance/{topic}'
            try:
                response = requests.get(url, headers=self.headers, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'figure']):
                        tag.decompose()
                    
                    article = soup.find('article') or soup.find('div', class_='article-content')
                    if article:
                        text = ' '.join([p.get_text() for p in article.find_all(['p', 'li', 'h2', 'h3'])])
                        text = re.sub(r'\s+', ' ', text).strip()
                        
                        if len(text) > 300:
                            self.documents.append({
                                'source': 'NerdWallet',
                                'url': url,
                                'text': text
                            })
                            print(f"✓ Scraped: {topic}")
                
                time.sleep(1.5)
            except Exception as e:
                print(f"✗ Failed {topic}: {str(e)}")
    
    def save_raw_documents(self, filepath='raw_documents.json'):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved {len(self.documents)} documents to {filepath}")

class RecursiveTextSplitter:
    def __init__(self, chunk_size=600, chunk_overlap=120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(self, text):
        chunks = []
        self._split_recursive(text, chunks, 0)
        return chunks
    
    def _split_recursive(self, text, chunks, sep_index):
        if len(text) <= self.chunk_size:
            if text.strip():
                chunks.append(text.strip())
            return
        
        if sep_index >= len(self.separators):
            self._split_by_chars(text, chunks)
            return
        
        separator = self.separators[sep_index]
        splits = text.split(separator) if separator else list(text)
        
        current_chunk = ""
        for split in splits:
            test_chunk = current_chunk + split + separator if current_chunk else split + separator
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                if len(split) > self.chunk_size:
                    self._split_recursive(split, chunks, sep_index + 1)
                    current_chunk = ""
                else:
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:].strip() + " " + split + separator
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
    def _split_by_chars(self, text, chunks):
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())

class KnowledgeBaseProcessor:
    def __init__(self, chunk_size=600, chunk_overlap=120):
        self.text_splitter = RecursiveTextSplitter(chunk_size, chunk_overlap)
        self.chunks = []
    
    def load_documents(self, filepath='raw_documents.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def clean_text(self, text):
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text.strip()
    
    def process_documents(self, documents):
        for doc in documents:
            cleaned_text = self.clean_text(doc['text'])
            splits = self.text_splitter.split_text(cleaned_text)
            
            for i, chunk in enumerate(splits):
                if len(chunk) > 100:
                    self.chunks.append({
                        'chunk_id': f"{doc['source']}_{len(self.chunks)}",
                        'text': chunk,
                        'source': doc['source'],
                        'url': doc['url'],
                        'chunk_index': i
                    })
        
        print(f"\n✓ Created {len(self.chunks)} chunks from {len(documents)} documents")
    
    def save_processed_chunks(self, filepath='knowledge_base_chunks.json'):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved chunks to {filepath}")
        
    def get_statistics(self):
        total_chars = sum(len(chunk['text']) for chunk in self.chunks)
        avg_chunk_size = total_chars / len(self.chunks) if self.chunks else 0
        
        sources = {}
        for chunk in self.chunks:
            sources[chunk['source']] = sources.get(chunk['source'], 0) + 1
        
        print("\n" + "="*50)
        print("=== Knowledge Base Statistics ===")
        print("="*50)
        print(f"Total Documents: {len(set(c['url'] for c in self.chunks))}")
        print(f"Total Chunks: {len(self.chunks)}")
        print(f"Average Chunk Size: {avg_chunk_size:.0f} characters")
        print(f"\nChunks by Source:")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            print(f"  {source}: {count}")
        print("="*50)

def main():
    print("="*50)
    print("Phase 1: Enhanced Data Curation & Preprocessing")
    print("="*50)
    
    scraper = FinancialKnowledgeBaseScraper()
    
    print("\n[1/4] Scraping CFPB...")
    scraper.scrape_cfpb()
    
    print("\n[2/4] Scraping Investopedia (Comprehensive)...")
    scraper.scrape_investopedia_comprehensive()
    
    print("\n[3/4] Scraping Investor.gov (Fixed)...")
    scraper.scrape_investor_gov_fixed()
    
    print("\n[4/4] Scraping NerdWallet...")
    scraper.scrape_nerdwallet()
    
    scraper.save_raw_documents()
    
    print("\n" + "="*50)
    print("Processing and Chunking Documents...")
    print("="*50)
    
    processor = KnowledgeBaseProcessor(chunk_size=600, chunk_overlap=120)
    documents = processor.load_documents()
    processor.process_documents(documents)
    processor.save_processed_chunks()
    processor.get_statistics()
    
    print("\n✓ Phase 1 Complete!")
    print("\nOutput Files:")
    print("  • raw_documents.json")
    print("  • knowledge_base_chunks.json")

if __name__ == "__main__":
    main()