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
from .mul_vector import Work
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
# from google.colab import userdata

# GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

genai.configure(api_key='AIzaSyBRXgwwPmyRTvspHzK6ozCPzxbAvrZHszQ')
genai_model = genai.GenerativeModel('gemini-pro')

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
        
        # 检查是否有有效的响应部分
        if hasattr(response, 'parts') and response.parts:
            return response.parts[0].text  # 假设返回的有效部分在第一个Part对象中
        else:
            # 检查是否因为安全原因阻止了内容
            if hasattr(response, 'candidate') and hasattr(response.candidate, 'safety_ratings'):
                return f"Content blocked due to safety filters: {response.candidate.safety_ratings}"
            else:
                return "No valid content was returned or content was blocked."
    except Exception as e:
        # 处理异常，返回错误信息
        return f"Error generating summary: {str(e)}"

def get_cosine_similarity(text1, text2):
    processed_text1 = " ".join(preprocess(text1))
    processed_text2 = " ".join(preprocess(text2))
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([processed_text1, processed_text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return cosine_sim[0][0]

def get_multidemensional_cosine_similarity(text1, text2):
    vector1 = Work(text1)
    vector2 = Work(text2)
    
    scaler = StandardScaler()
    vector1_scaled = scaler.fit_transform(vector1)
    vector2_scaled = scaler.fit_transform(vector2)

    # 计算标准化后的余弦相似度
    cosine_sim = cosine_similarity(vector1_scaled, vector2_scaled)
    
    return cosine_sim[0][0]

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
        
        words_set1 = get_top_words_distribution(document_topics_keywords[0])
        words_set2 = get_top_words_distribution(document_topics_keywords[1])
        
        common_words = words_set1.intersection(words_set2)
        
        summary = GeminiSummary(str(common_words))
        
        result = {'document':document_topics_keywords, 'corpus':corpus_topics_keywords, 'similarity':similarity, 'summary':summary}

        return JsonResponse(result)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)