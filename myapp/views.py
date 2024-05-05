from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
import gensim
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import LdaModel

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    processed_tokens = []
    
    for word in tokens:
        if word.lower() not in stop_words and word.isalnum():
            lemmatized_word = lemmatizer.lemmatize(word)
            processed_tokens.append(lemmatized_word)
            
    return processed_tokens

# def core(text1, text2):
#     texts = [preprocess(text1), preprocess(text2)]
#     dictionary = corpora.Dictionary(texts)
#     corpus = [dictionary.doc2bow(text) for text in texts]
    
#     num_topics = 10

#     lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    
#     corpus_topics_keywords = {}
#     for topic_id in range(num_topics):
#         top_words = lda_model.show_topic(topic_id, topn=50)
#         corpus_topics_keywords[topic_id] = {word: float(weight) for word, weight in top_words}

#     document_topics_keywords = []
    
#     for bow in corpus:
#         doc_topics = lda_model.get_document_topics(bow, minimum_probability=0)
#         doc_keywords = {}

#         for topic_id, prob in doc_topics:
#             doc_keywords[topic_id] = {
#                 'prob':float(prob),
#                 'words':{}
#             } 
#             top_words = lda_model.show_topic(topic_id, topn=50)
#             doc_keywords[topic_id]['words'] = {word: float(weight) for word, weight in top_words}

#         document_topics_keywords.append(doc_keywords)
        
    
        
#         print("text1:",document_topics_keywords[0])
#         print("------------------------------------------------------------------")
#         print("text2:",document_topics_keywords[1])
    
#     return document_topics_keywords, corpus_topics_keywords
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
                'words': {}
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


def get_cosine_similarity(text1, text2):
    processed_text1 = " ".join(preprocess(text1))
    processed_text2 = " ".join(preprocess(text2))
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([processed_text1, processed_text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return cosine_sim[0][0]

@csrf_exempt
@require_http_methods(["POST"])
def compare_texts(request):
    try:
        data = json.loads(request.body)
        text1 = data.get('text1')
        text2 = data.get('text2')
        
        document_topics_keywords, corpus_topics_keywords = core(text1, text2)
        similarity = get_cosine_similarity(text1, text2)
#         similarity = 100
        
        result = {'document':document_topics_keywords, 'corpus':corpus_topics_keywords, 'similarity':similarity}

        return JsonResponse(result)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)