import re
import os
import json
import random
import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import yaml
from sklearn.manifold import TSNE
import ast
from sklearn.feature_extraction.text import CountVectorizer


from topic_model_util import generate_random_color, generate_strong_red_color, take_dict_value

# Returns
def get_representative_text_idex(topic_idx, representative_doc_dict, df_merge):
    representative_text_list = representative_doc_dict[topic_idx]
    representative_text_idx_list = df_merge.index[df_merge['text'].isin(representative_text_list)].tolist()
    return representative_text_idx_list


def string_to_list(string):
    # Safely evaluate the string as a Python literal
    evaluated = ast.literal_eval(string)

    # Ensure the evaluated value is a list
    if isinstance(evaluated, list):
        return evaluated
    else:
        return []

# Assigning a grey color to outlier topic, and red colors to negative topics.
# Otherwise random colors outside red color.
def assign_topic_color(topic_idx, negative_topic_idx_list):
    if topic_idx == -1:
        return 128, 128, 128
    elif topic_idx in negative_topic_idx_list:
        return generate_strong_red_color()
    else:
        return generate_random_color()

def export_dash_data(df_merge, df_topic, topic_model):
    elm_list = [
        {'data': {
            'id': str(idx),
            'topic_idx': row.Topic,
            'color': 'rgb{}'.format(row.topic_color),
            'text': row.text
        },
            'position': {
                'x': float(row.x),
                'y': float(row.y)
            },
            'selectable': True,
            'grabbable': False
        } for idx, row in df_merge.iterrows()
    ]

    topic_dict = {
        key: ['{}*"{}"'.format(temp_tuple[1], temp_tuple[0]) for temp_tuple in value] for key, value in
        topic_model.get_topics().items()
    }

    df_merge.to_csv("../topic_modeling_data/topic_model_df.csv")

    with open('../topic_modeling_data/job_topic_elm_list.json', 'w') as f:
        json.dump({'elm_list': elm_list}, f)

    with open('../topic_modeling_data/job_topics.json', 'w') as f:
        json.dump(topic_dict, f)

    topic_color_list = df_topic.groupby(['Topic', 'topic_color']).sum().reset_index()[['Topic', 'topic_color']][
        'topic_color'].tolist()

    topic_color_list = ['rgb{}'.format(elem) for elem in topic_color_list]

    with open('../topic_modeling_data/topic_color.txt', 'w') as f:
        for d in topic_color_list:
            f.write("%s\n" % d)



def generate_BERTopic_cytoscape(df_data):
    # Removing duplicates of texts
    df_corpus = df_data[~df_data.duplicated(subset=['text'])].iloc[1:, :].reset_index(drop=True)

    # Using the raw scraped texts.
    corpus = df_corpus['text']

    # Prepare embeddings with a pre-trained a Transformer model.
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(corpus, show_progress_bar=True)

    # Train BERTopic
    # Setting nr_topics to 'auto' enables somewhat reducing number of topics.
    topic_model = BERTopic(nr_topics="auto").fit(corpus, embeddings) # note that UMAP is used automatically by this function

    # Removes stop words from topic representations and get updated topics
    # *If you comment out two lines below, you get topics full of stopwords such as "a" or "and" or "of"
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 5))
    topic_model.update_topics(corpus, vectorizer_model=vectorizer_model)

    # Getting information of inferred topics by topic modeling.
    df_document_info = pd.DataFrame(topic_model.get_document_info(corpus))

    # Merging topic indexes to the original texts, row by row
    df_merge = pd.merge(df_corpus, df_document_info['Topic'],
                        left_index=True,
                        right_index=True)

    # Reducing dimensions of embeddings hundreds of dimensions to two dimensions for visualization.
    dim_red_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine')
    reduced_embeddings = dim_red_model.fit_transform(embeddings, y=df_merge['Topic'].tolist())

    # Merging information of x, y coordinates of each text
    df_merge[['x', 'y']] = pd.DataFrame(reduced_embeddings)

    # Calculating sentiment scores topic by topic.
    df_topic_sentiment = df_merge.groupby('Topic').agg(
        {'sentiment_neutral': 'mean', 'sentiment_pos': 'mean', 'sentiment_neg': 'mean'}
    )
    df_topic_sentiment = df_topic_sentiment.reset_index()

    # Merging information of ocurrences of each topic
    freq = topic_model.get_topic_info()
    df_freq = pd.DataFrame(freq)
    df_topic = pd.merge(df_freq, df_topic_sentiment, how='inner', on='Topic')

    # Bertopic finds the representative texts on each topic, and the part below get indexes of those texts.
    representative_doc_dict = topic_model.get_representative_docs()
    df_topic['representative_text_indexes'] = df_topic['Topic'].apply(
        lambda x: df_merge.index[df_merge['text'].isin(representative_doc_dict[x])].tolist())

    # When negative sentiment scores surpass that of positive sentiment, the topic is detected as negative.
    # And we give random red colors to those alarming topics, otherwise random other colors.
    negative_topic_idx_list = df_topic_sentiment.query("sentiment_pos < sentiment_neg").Topic.tolist()
    df_topic['topic_color'] = df_topic['Topic'].apply(lambda x: assign_topic_color(x, negative_topic_idx_list))
    df_merge = df_merge.merge(df_topic[['Topic', 'topic_color']], on='Topic', how='inner')

    return df_topic, df_merge, topic_model



if __name__ == '__main__':
    # Read scraped and preprocessed data.
    df_data = pd.read_csv('../data/data_to_export.csv')

    # Apply BERTopic on the scraped data
    # df_topic: information of detected topic
    # df_merge: scraped information merged with topic and sentiment data
    # topic_model: BERTopic model that is optimized on our scraped data.
    df_topic, df_merge, topic_model = generate_BERTopic_cytoscape(df_data)

    # Saves visualization for Dash
    export_dash_data(df_merge, df_topic, topic_model)

    # Saves resutls as csv files.
    df_topic.to_csv('../data/bertopic_topic.csv', encoding='utf-8', index=False)
    df_merge.to_csv('../data/bertopic_result.csv', encoding='utf-8', index=False)