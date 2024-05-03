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

def core(text1, text2):
    texts = [preprocess(text1), preprocess(text2)]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    num_topics = 10

    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    
    corpus_topics_keywords = {}
    for topic_id in range(num_topics):
        top_words = lda_model.show_topic(topic_id, topn=50)
        corpus_topics_keywords[topic_id] = {word: float(weight) for word, weight in top_words}
#         整个语料库的每个主题的前五十个高权重词
#         top_wrods有word, prob两个属性

    document_topics_keywords = []
#     每个文档中的每个主题所包含的每个单词
    
    for bow in corpus:
        doc_topics = lda_model.get_document_topics(bow, minimum_probability=0)
        doc_keywords = {}

        for topic_id, prob in doc_topics:
            doc_keywords['prob'] = float(prob)
#             每个主题的权重
            top_words = lda_model.show_topic(topic_id, topn=50)
            doc_keywords[topic_id] = {word: float(weight) for word, weight in top_words}

        document_topics_keywords.append(doc_keywords)
    
    return document_topics_keywords, corpus_topics_keywords

@csrf_exempt
@require_http_methods(["POST"])
def compare_texts(request):
    try:
        data = json.loads(request.body)
        text1 = data.get('text1')
        text2 = data.get('text2')
        
        document_topics_keywords, corpus_topics_keywords = core(text1, text2)
        
        result = {'document':document_topics_keywords, 'corpus':corpus_topics_keywords}

        return JsonResponse(result)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)