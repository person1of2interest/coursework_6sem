import pandas as pd
import os
import numpy as np
import random
import math
from functools import reduce
from sklearn.cluster import KMeans #для сравнения с версией, написанной вручную


DATA_PATH = '../data/'

#формируем dataframe из csv-файла
#если 'ratings.csv' находится в той же папке, 
#можно просто pd.read_csv('ratings.csv', ...)
ratings_main = pd.read_csv(
    os.path.join(DATA_PATH, 'ratings.csv'),
    dtype={
        'user_uid': np.uint32,
        'element_uid': np.uint16,
        'rating': np.uint8,
        'ts': np.float64,
    }
)

#генерируем словарь словарей
def generate_ratings_dict(data) -> dict:
    ratings_dict = {}
    
    for _, item in data.iterrows():
        user_id = item['user_uid']
        
        if user_id not in ratings_dict:
            ratings_dict[user_id] = {} #ключи - идентификаторы пользователей
        
        #значения - словари вида 'идентификатор товара' : оценка
        (ratings_dict[user_id])[item['element_uid']] = item['rating']
    
    return ratings_dict

#генерируем словарь оценок
def generate_data(ratings_dict: dict, movies_list: list) -> dict:
    data_dict = {}
    n = len(movies_list)
    
    for user_id, item in ratings_dict.items():        
        if user_id not in data_dict:
            data_dict[user_id] = [0]*n #ключи - идентификаторы пользователей
        
        for i in range(n):
            if movies_list[i] in ratings_dict[user_id]:
                (data_dict[user_id])[i] = (ratings_dict[user_id])[movies_list[i]]
                #значения - списки оценок пользователем товаров, одинаковой длины
                #если пользователь не оценивал товар стоит 0

    return data_dict

#выбираем произвольных k пользователей в качестве начальных центроидов
def generate_centroids(k: int, data_list: list) -> list: 
    return random.sample(data_list, k)

#расстояние между двумя векторами
def distance(u: list, v: list) -> float: 
    dist = 0

    for i in range(len(u)):
        dist += math.pow((u[i] - v[i]), 2)

    return math.sqrt(dist)

#определяем для каждой точки кластер
def add_to_cluster(item: int, centroids: list, data_dict: dict):
    return item, min(range(len(centroids)), key = lambda i: distance(data_dict[item], centroids[i]))

#сумма векторов
def add_vector(u: list, v: list) -> list:
    return [u[k] + v[k] for k in range(len(v))]

def move_centroids(k: int, iteration: list, data_dict: dict) -> list:
    centroids = []

    for cen in range(k):
        members = [i[0] for i in iteration if i[1] == cen] #список всех точек кластера

        if members:
            centroid = [i/len(members) for i in reduce(add_vector, [data_dict[m] for m in members])]
            centroids.append(centroid)
    
    return centroids

def k_means(k: int, data_dict: dict) -> list:
    best_weight = math.inf
    data_list = list(data_dict)
    centroids = [data_dict[i] for i in generate_centroids(k, data_list)]

    while True:
        iteration = list([add_to_cluster(item, centroids, data_dict) for item in data_list])
        new_weight = 0
        
        for i in iteration:
            new_weight += distance(data_dict[i[0]], centroids[i[1]]) 
            #прибавляем расстояние между каждой точкой и ее центроидом
        
        #если новый вес меньше, чем лучший, продолжаем
        #иначе возвращаем кластеризованных пользователей
        if new_weight < best_weight:
            best_weight = new_weight
            new_weight = 0
        else:
            return iteration

        centroids = move_centroids(k, iteration, data_dict)

#запускаем на наших данных
#вырезаем первые 2000 строк для тестирования
ratings = ratings_main.iloc[:2000]

#получаем список всех фильмов, которые хоть кто-то оценил
movies_list = list(set(ratings['element_uid'])) 
ratings_dict = generate_ratings_dict(ratings)
data_dict = generate_data(ratings_dict, movies_list)
clustered_users = k_means(4, data_dict) #получаем список кортежей вида (пользователь, номер кластера)

#проделываем то же с помощью библиотеки scikit-learn
X = list(data_dict.values())

km = KMeans(
    n_clusters=4, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)
clustered_sklearn = [(list(data_dict)[i], y_km[i]) for i in range(len(y_km))]

#сравнивам результаты
symm_diff = set()
intersection = set()
clustered_users = sorted(clustered_users, key = lambda t : t[0])
clustered_sklearn = sorted(clustered_sklearn, key = lambda t : t[0])

#так как кластеры могут нумероваться в любом порядке, сравним кластеры, 
#соответствующие первому пользователю в отсортированном списке
num_1 = clustered_users[0][1]
num_2 = clustered_sklearn[0][1]

for i in range(len(clustered_users)):
    if (clustered_users[i][1] == num_1 and clustered_sklearn[i][1] != num_2) or (clustered_users[i][1] != num_1 and clustered_sklearn[i][1] == num_2):
        symm_diff.add(clustered_users[i])
    elif clustered_users[i][1] == num_1 and clustered_sklearn[i][1] == num_2:
        intersection.add(clustered_users[i])

print(format(100*len(symm_diff)/len(intersection), '.2f')) 
#выводится отношение (в %) симметрической разности 
#множеств пользователей для этих двух кластеров к их пересечению