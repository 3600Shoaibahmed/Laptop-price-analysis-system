#!pip install nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

class SimpleRetrievalSystem:
    def __init__(self, documents):
        self.documents = documents

    def preprocess_text(self, text):
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token.isalnum() and token not in stop_words]

    def retrieve_documents(self, query):
        query_words = set(self.preprocess_text(query))
        retrieved_documents = []
        for i, doc in enumerate(self.documents):
            doc_words = set(self.preprocess_text(doc))
            if query_words.intersection(doc_words):
                retrieved_documents.append(f"doc{i+1}")
        return retrieved_documents

    def precision(self, retrieved, relevant):
        return len(set(retrieved) & set(relevant)) / len(retrieved) if len(retrieved) > 0 else 0

    def recall(self, retrieved, relevant):
        return len(set(retrieved) & set(relevant)) / len(relevant) if len(relevant) > 0 else 0

    def f1_score(self, precision, recall):
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def fallout(self, retrieved, irrelevant):
        return len(set(retrieved) & set(irrelevant)) / len(irrelevant) if len(irrelevant) > 0 else 0

# Sample documents and relevance judgments
documents = ["This is document 1.", "Document 2 is different.", "Document 3 contains important information.", "Another document here.", "This is the last document."]

# Corrected relevant documents
relevant_documents = {"document": ['doc1', 'doc2', 'doc3', 'doc4', 'doc5'], "information": ['doc3'], "important": ['doc5']}

# Initialize the retrieval system
retrieval_system = SimpleRetrievalSystem(documents)

# Sample queries
queries = ["document", "information", "important"]

# Evaluate each query
for query in queries:
    # Retrieve documents using the retrieval system
    retrieved_documents = retrieval_system.retrieve_documents(query)
    # Get the relevant documents for the query
    relevant_docs = relevant_documents.get(query, [])
    # Get the irrelevant documents for the query
    all_docs = [f"doc{i+1}" for i in range(len(documents))]
    irrelevant_docs = list(set(all_docs) - set(relevant_docs))
    # Calculate and print evaluation metrics
    print(f"Query: {query}")
    print(f"Retrieved Documents: {retrieved_documents}")
    print(f"Precision: {retrieval_system.precision(retrieved_documents, relevant_docs)}")
    print(f"Recall: {retrieval_system.recall(retrieved_documents, relevant_docs)}")
    print(f"F1 Score: {retrieval_system.f1_score(retrieval_system.precision(retrieved_documents, relevant_docs), retrieval_system.recall(retrieved_documents, relevant_docs))}")
    print(f"Fallout: {retrieval_system.fallout(retrieved_documents, irrelevant_docs)}")
    print("------")




