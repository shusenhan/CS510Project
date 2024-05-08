from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
import gensim
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import LdaModel
from .mul_vector import Work
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
import numpy as np
from scipy.sparse import csr_matrix
from .plsa import plsa

genai.configure(api_key='AIzaSyBRXgwwPmyRTvspHzK6ozCPzxbAvrZHszQ')
genai_model = genai.GenerativeModel('gemini-1.5-pro-latest')

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def preprocess(text):
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    tokens = word_tokenize(text)

    processed_tokens = []
    for word in tokens:
        if word not in stop_words and word.isalnum():
            lemmatized_word = lemmatizer.lemmatize(word)
            processed_tokens.append(lemmatized_word)

    return processed_tokens

def docuement_analyze(text):
    texts = [preprocess(text)]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    num_topics = 5
    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    document_topics_keywords = []
    
    for bow in corpus:
        doc_topics = lda_model.get_document_topics(bow, minimum_probability=0)
        doc_keywords = {}

        for topic_id, prob in doc_topics:
            doc_keywords[topic_id] = {
                'prob': float(prob),
                'words': {},
                'summary': "",
            }
            top_words = lda_model.show_topic(topic_id, topn=20)
            doc_keywords[topic_id]['words'] = {word: float(weight) for word, weight in top_words}

        document_topics_keywords.append(doc_keywords)
    
    return document_topics_keywords[0]


def core(text1, text2):
    texts = [preprocess(text1), preprocess(text2)]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    num_topics = 5
    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    
    corpus_topics_keywords = {}
    for topic_id in range(num_topics):
        top_words = lda_model.show_topic(topic_id, topn=20)
        corpus_topics_keywords[topic_id] = {word: float(weight) for word, weight in top_words}

    document_topics_keywords = []
    
    document_topics_keywords.append(docuement_analyze(text1))
    document_topics_keywords.append(docuement_analyze(text2))
    
    return document_topics_keywords, corpus_topics_keywords

def GeminiSummary(text):
    try:
        prompt = "This is high-wieght common words set in two documents, make a brief sentence (no more than 20 words) to summarize a possible topic: "+ text
        response = genai_model.generate_content(prompt)
        
        if hasattr(response, 'parts') and response.parts:
            return response.parts[0].text
        else:
            if hasattr(response, 'candidate') and hasattr(response.candidate, 'safety_ratings'):
                return f"Content blocked due to safety filters: {response.candidate.safety_ratings}"
            else:
                return "No valid content was returned or content was blocked."
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def get_cosine_similarity(text1, text2):
    processed_text1 = " ".join(preprocess(text1))
    processed_text2 = " ".join(preprocess(text2))
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([processed_text1, processed_text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return cosine_sim[0][0]

def pearson_correlation(text1, text2):
    processed_text1 = " ".join(preprocess(text1))
    processed_text2 = " ".join(preprocess(text2))

    tfidf_vectorizer = TfidfVectorizer(min_df=1, max_df=1.0, max_features=2000)
    tfidf_matrix = tfidf_vectorizer.fit_transform([processed_text1, processed_text2])
    
    dense_matrix = tfidf_matrix.toarray()
    vector1 = dense_matrix[0]
    vector2 = dense_matrix[1]

    mean_vector1 = np.mean(vector1)
    mean_vector2 = np.mean(vector2)

    numerator = np.sum((vector1 - mean_vector1) * (vector2 - mean_vector2))
    denominator = np.sqrt(np.sum((vector1 - mean_vector1) ** 2)) * np.sqrt(np.sum((vector2 - mean_vector2) ** 2))

    if denominator == 0:
        return 0
    correlation = numerator / denominator
    return correlation

def bert_similarity(doc1, doc2):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

    encoding = tokenizer(doc1, doc2, return_tensors='pt', padding=True, truncation=True, truncation_strategy='only_first')

    outputs = model(**encoding)
    logits = outputs.logits
    similarity_score = torch.sigmoid(logits[0, 0]).item()

    return similarity_score

def get_top_words_distribution(document):
    top_three_topics = sorted(document.items(), key=lambda x: x[1]['prob'], reverse=True)[:3]
    
    words_set=set()
    
    for topic_id, info in top_three_topics:
        top_words = sorted(info['words'].items(), key=lambda x: x[1], reverse=True)[:6]
        words_set.update(word for word, _ in top_words)

    return words_set
    

@csrf_exempt
@require_http_methods(["POST"])
def compare_texts(request):
    try:
        data = json.loads(request.body)
        text1 = data.get('text1')
        text2 = data.get('text2')
        
        document_topics_keywords, corpus_topics_keywords = core(text1, text2)
        similarity = get_cosine_similarity(text1, text2)
        bert_sim = bert_similarity(text1,text2)
        correlation = pearson_correlation(text1,text2)
        
        words_set1 = get_top_words_distribution(document_topics_keywords[0])
        words_set2 = get_top_words_distribution(document_topics_keywords[1])
        
        common_words = words_set1.intersection(words_set2)
        
        summary = GeminiSummary(str(common_words))
        
        plsa_matrix = plsa(text1, text2)
        
        result = {'document':document_topics_keywords, 'corpus':corpus_topics_keywords, 'similarity':similarity, 'summary':summary, 'bert_sim':bert_sim, 'correlation':correlation, 'plsa':plsa_matrix}

        return JsonResponse(result)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)