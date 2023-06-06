import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
import random


def normalize(x):
    x = x.astype(float)
    x_sum = x.sum()
    x_num = x.astype(bool).sum()
    if x_num == 0:
        print(x)
    x_mean = x_sum / x_num
    x_range = x.max() - x.min()

    if x_range == 0:
        return 0.0
    
    return (x - x_mean) / x.std()


def get_coo_matrix(ratings_main):
    
    # составляю разреженную матрицу оценок пользователями просмотренных фильмов
    # по датасету ratings_main
    # строки соответствуют пользователям, столбцы --- фильмам

    users = ratings_main['user_uid'].astype('category')
    elements = ratings_main['element_uid'].astype('category')
    ratings = ratings_main.groupby('user_uid')['rating'].transform(lambda x: normalize(x))
    
    coo = coo_matrix((ratings.astype(float),
                    (users.cat.codes.copy(), elements.cat.codes.copy())))
    
    return coo


def get_cor_matrix(ratings_main):
    
    coo = get_coo_matrix(ratings_main)
    
    # считаю матрицу перекрытия, которая показывает,
    # сколько пользователей оценили оба фильма i и j
    overlap_matrix = coo.T.astype(bool).astype(int) @ coo.astype(bool).astype(int) 
    
    # задаю минимальное допустимое число пользователей, оценивших оба фильма
    min_overlap = 3
    
    # вычисляю матрицу сходства фильмов
    cor = cosine_similarity(coo.T, dense_output=False)
    cor = cor.multiply(cor > 0.4) # удаляю слишком низкие значения сходств
    
    # удаляю сходства с недостаточным перекрытием
    cor = cor.multiply(overlap_matrix > min_overlap) 
    
    return cor


def predict_rating(user, elem, ratings_main, coo, cor):
    
    mean_r = ratings_main.groupby('user_uid')['rating'].mean()[user]
    i = ratings_main[ratings_main['user_uid']==user].index[0]
    j = ratings_main[ratings_main['element_uid']==elem].index[0]
    ind1 = ratings_main['user_uid'].astype('category').cat.codes.to_frame().loc[i]
    ind2 = ratings_main['element_uid'].astype('category').cat.codes.to_frame().loc[j]
    numerator = coo.toarray()[ind1[0]] @ cor.toarray()[ind2[0]]
    denominator = cor.toarray()[ind2[0]].sum()
    
    if denominator == 0:
        return 0.0
    
    return mean_r + numerator / denominator

def predict(user, ratings_main):
    ratings_without_user = ratings_main.drop(labels=ratings_main.groupby('user_uid').get_group(user).index, 
                                         inplace=False)
    elements_list = list(ratings_without_user['element_uid'].unique())
    elements_sample = random.sample(elements_list, k=20)
    predicted_ratings = {}
    
    coo = get_coo_matrix(ratings_main)
    cor = get_cor_matrix(ratings_main)

    for elem in elements_sample:
        r = predict_rating(user, elem, ratings_main, coo, cor)
        predicted_ratings[elem] = round(r, 2)
    
    return predicted_ratings


def main():
    DATA_PATH = '../data/'

    ratings_main = pd.read_csv(
        os.path.join(DATA_PATH, 'ratings.csv'),
        dtype={
            'user_uid': np.uint32,
            'element_uid': np.uint16,
            'rating': np.uint8,
            'ts': np.float64,
        }
    )


    # удаляю из списка фильмов фильмы с одной единственной оценкой на весь датасет
    data = ratings_main.groupby('element_uid').agg({'element_uid': ['count']})
    ind = set(data[data['element_uid']['count'] > 2].index)
    grouped = ratings_main.groupby('element_uid')
    ratings_main = grouped.filter(lambda x: set(x['element_uid']).issubset(ind))
    
    # удаляю строки с оценкой фильма 0 (используется шкала от 1 до 10, поэтому это мусор)
    ratings_main.drop(labels=ratings_main.groupby('rating').get_group(0).index, 
                      inplace=True)

    # отбираю 30_000 строк для обучения 
    # с помощью стратифицированной по элементам выборки
    sss = StratifiedShuffleSplit(n_splits=1, train_size=30_000, random_state=0)
    for _, (train_index, _) in enumerate(sss.split(ratings_main, ratings_main['element_uid'])):
        ratings_main = ratings_main.iloc[train_index]

    # пример составления рекомендаций для пользователя 
    user = 72
    print(predict(user, ratings_main))

