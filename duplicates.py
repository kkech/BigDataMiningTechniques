from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Duplicates:
    def findDuplicates(data):
        #     1. calculate tf-idf for the document
        #     2. calculate cosine similarity for two given text
        #     3. the cosine similarity will indicate match between two documents.
        # vectorize document content TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')  # Convert a collection of raw documents to a matrix of TF-IDF features.
        X = vectorizer.fit_transform(data['Content'])  # Fit to data, then transform it.
        cosine_similarities = cosine_similarity(X, X)
        return cosine_similarities