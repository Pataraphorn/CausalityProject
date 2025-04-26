from .._utils import *
from . import bert, top2vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class topicModel:
    def __init__(self, topic_type="BERTopic"):
        super().__init__()
        self.model = None
        self.topic_type = topic_type
        self.topic_dict = None
        self.df = pd.DataFrame()
        self.data_col = "content"
        print("Processing with topic modeling name:", self.topic_type)
        if self.topic_type == 'BERTopic':
            print(f'Start Initialized BERTopic')
        elif self.topic_type == 'Top2Vec':
            print(f'Start Initialized Top2Vec')
        else:
            print(f'We do not have the {topic_type} mode yet.')

    def set_data_col(self, data_col: str):
        self.data_col = data_col

    def topic_from_trained_model(self, model_path: str, embedding_model: str ="all-MiniLM-L6-v2"):    
        if self.topic_type == 'BERTopic':
            self.model = bert.load_model(path_to_load=model_path, model_embedding_name=embedding_model)
            self.topic_dict = self.get_dict_of_topic()
            print('Finish getting dictionary of topics')
        elif self.topic_type == 'Top2Vec':
            self.model = top2vec.load_model(path_to_load=model_path)
            self.topic_dict = self.get_dict_of_topic()
            print('Finish getting dictionary of topics')
        else:
            print(f'We do not have the {self.topic_type} mode yet.')

    def training_model(
        self, df: pd.DataFrame, embedding_path: str = None, batch_size: int = 500, model_embedding_name: str ='all-MiniLM-L6-v2'
    ):
        if self.topic_type == 'BERTopic':
            if embedding_path is None:
                embedding_path = bert.run_embedding(df, data_col=self.data_col, batch_size=batch_size, model_embedding_name=model_embedding_name)
            self.model = bert.update_model(
                df, data_col=self.data_col, embeddings_load=embedding_path
            )
            self.topic_dict = self.get_dict_of_topic()
            # self.topic_pos = BERT.get_coor_topic(self)
            print(f"Finish training {self.topic_type} model of topics")
        else:
            print(f'We do not have the {self.topic_type} mode yet.')

    def save_model(self, model_path: str, embedding_model: str ="all-MiniLM-L6-v2"):
        if self.topic_type == 'BERTopic':
            bert.save_model(self.model, path_to_save=model_path, embedding_model=embedding_model)
        else:
            print(f'We do not have the {self.topic_type} mode yet.')

    def get_topic(self, df: pd.DataFrame, map: dict = None):
        if map:
            df["topic"] = df[self.data_col].progress_apply(lambda x: map(x))
        elif self.topic_type == 'BERTopic':
            df["topic"] = self.model.topics_
        else:
            print(f'We do not have the {self.topic_type} mode yet.')
        return df

    def get_dict_of_topic(self):
        if self.topic_type == 'BERTopic':
            topics_name_dict = self.model.get_topic_info().set_index('Topic')['Name'].T.to_dict()
            # topics_name_dict[-2] = '<PAD>'
            print(f"Number of topics : {len(topics_name_dict)}")
            return topics_name_dict
        else:
            print(f'We do not have the {self.topic_type} mode yet.')
            return None

    def get_embeddings(self):
        if self.topic_type == 'BERTopic':
            all_topics = sorted(list(self.model.get_topics().keys()))
            freq_topic = self.model.get_topic_freq()
            freq_topic = freq_topic.loc[freq_topic.Topic != -1, :]
            topics = sorted(freq_topic.Topic.to_list())
            indices = np.array([all_topics.index(topic) for topic in topics])
            return self.model.topic_embeddings_[indices]
        else:
            print(f'We do not have the {self.topic_type} mode yet.')
            return None

    def get_corr_topic(self):
        if self.topic_type == 'BERTopic':
            corr = bert.get_coor_topic(self.model)
            return corr
        else:
            print(f'We do not have the {self.topic_type} mode yet.')
            return None

    def topic_diversity(self, top_n=20):
        # Get topic words
        topics = self.model.get_topics()
        top_n = 10

        # Extract words from topics
        topic_words = [[word for word, _ in topics[topic]] for topic in topics if topic != -1]

        # Flatten and check uniqueness
        unique_words = set(word for topic in topic_words for word in topic[:top_n])
        diversity = len(unique_words) / (len(topic_words) * top_n)
        return diversity

    def topic_coherence(self, top_n=10, window_size=20):
        topic_words = self.model.get_topics()
        topic_words = [ [w for w, _ in topic_words[k]] for k in topic_words if k != -1 ]
        
        docs = self.df[self.data_col].tolist()

        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(docs)
        terms = vectorizer.get_feature_names_out()
        
        term_index = {term: idx for idx, term in enumerate(terms)}
        topic_coherences = []

        for topic in topic_words:
            topic = [w for w in topic if w in term_index]
            if len(topic) < 2:
                topic_coherences.append(0)
                continue

            indices = [term_index[w] for w in topic]
            submatrix = X[:, indices].toarray()
            co_matrix = cosine_similarity(submatrix.T)
            
            # Take the upper triangle without the diagonal
            upper = np.triu_indices_from(co_matrix, k=1)
            coherence = co_matrix[upper].mean()
            topic_coherences.append(coherence)

        return np.mean(topic_coherences)