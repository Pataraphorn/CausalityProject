from .._utils import *
import torch

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.preprocessing import MinMaxScaler
import gc

if torch.cuda.is_available():
    from cuml.cluster import HDBSCAN
    from cuml.manifold import UMAP
else :
    from hdbscan import HDBSCAN
    from umap import UMAP

def create_umap_hdbscan_models():
    umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0, low_memory=True)
    hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True)
    return umap_model, hdbscan_model

def load_model(path_to_load: str, model_embedding_name: str = 'all-MiniLM-L6-v2'):
    model_embedding = SentenceTransformer(model_embedding_name)
    return BERTopic.load(path_to_load, embedding_model=model_embedding)

def run_embedding(df: pd.DataFrame, data_col: str='content', save_path: str=f'{current_dir}/embeddings', model_embedding_name:str='all-MiniLM-L6-v2', batch_size:int=1):
    print('Start Embedding')
    model_embedding = SentenceTransformer(model_embedding_name)
    embeddings = model_embedding.encode(df[data_col].to_list(),
                                    show_progress_bar=True,
                                    batch_size=batch_size)
    np.savez_compressed(save_path, embeddings=embeddings)
    gc.collect()
    print(f"Save Embedding to {save_path}.npz")
    return save_path+'.npz'


def update_model(df:pd.DataFrame, data_col:str ='content', batch_size = 25000, embeddings_load=None):
    print(f"check embeddings path:{embeddings_load}")
    if embeddings_load is not None:
        print(f'Loading pretrained embedding from path:{embeddings_load}')
        embeddings = np.load(embeddings_load)['embeddings']
    else:
        run_embedding(df)
        embeddings = np.load(f'{current_dir}/embeddings')['embeddings']

    batch = len(df) // batch_size

    print('Start creating a embedding model')
    bert = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model)

    for i in tqdm(range(batch)):
        umap_model, hdbscan_model = create_umap_hdbscan_models()
        start = i*batch_size
        stop = min((i + 1) * batch_size, len(df))

        print(f'Started from {start} to {stop} amount:{len(df[start:stop])} docs')
        if i == 0:
            base_model = bert.fit(
                df[start:stop][data_col].tolist(), embeddings[start:stop]
            )
        elif i > 0:
            new_model = bert.fit(
                df[start:stop][data_col].tolist(), embeddings[start:stop]
            )
            base_model = BERTopic.merge_models([base_model, new_model])
            gc.collect()

    print('Updating topics with n_gram_range=(1, 2)')
    base_model.update_topics(df[data_col].tolist(), n_gram_range=(1, 2))
    gc.collect()
    return base_model

def save_model(topic_model:BERTopic, path_to_save:str, embedding_model:str):
    print(f'Saving BERT model with {embedding_model} embedding model')
    topic_model.save(path_to_save, serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)


def get_coor_topic(topic_model):
    print("Stating get coordinates of each topic")
    all_topics = sorted(list(topic_model.get_topics().keys()))
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    topics = sorted(freq_df.Topic.to_list())
    indices = np.array([all_topics.index(topic) for topic in topics])
    embeddings = topic_model.c_tf_idf_.toarray()[indices]
    print("Warning: This may take a lot of time.")
    embeddings = MinMaxScaler().fit_transform(embeddings)
    embeddings = UMAP(
        n_neighbors=2, n_components=2, metric="hellinger", random_state=42
    ).fit_transform(embeddings)
    coor = {k: v for (k, v) in zip(topics, embeddings)}
    return coor
