import pandas as pd
import os
import numpy as np
from collections import defaultdict
from itertools import combinations
import json


# генерируем словарь всех транзакций
def generate_transactions(data) -> dict:
    transactions = dict()
    
    for _, trans_item in data.iterrows():
        user_id = trans_item['user_uid']
        
        if user_id not in transactions:
            transactions[user_id] = []	# ключи - идентификаторы пользователей
        
        # значения - списки идентификаторов фильмов 
        # из списка транзакций соответствующего пользователя
        transactions[user_id].append(trans_item['element_uid']) 
        # то есть не учитываем тип потребления
    
    return transactions


# генерируем словарь одноэлементных множеств
def calculate_itemsets_one(transactions_dict: dict, min_sup=0.005) -> dict:
    N = len(transactions_dict)
    temp_dict = defaultdict(int)
    one_itemsets = {}
    
    for key, items in transactions_dict.items():
        for item in items:
            inx = frozenset({item}) 
            temp_dict[inx] += 1

    for key, itemset in temp_dict.items():
        if itemset > min_sup * N:	# не включаем неподдерживаемые элементы
            one_itemsets[key] = itemset
    
    return one_itemsets


def has_support(items: list, one_itemsets: dict) -> bool:
    return ((frozenset({items[0]}) in one_itemsets) and (frozenset({items[1]}) in one_itemsets))


# генерируем словарь двухэлементных множеств
def calculate_itemsets_two(transactions_dict: dict, one_itemsets: dict) -> dict:
    two_itemsets = defaultdict(int)
    
    for key, items in transactions_dict.items():
        items = list(set(items)) # удаляем повторения
        
        if len(items) > 2:
            for perm in combinations(items, 2):	# рассматриваем все сочетания по 2 элемента
                if has_support(perm, one_itemsets):
                    two_itemsets[frozenset(perm)] += 1
        elif len(items) == 2:
            if has_support(items, one_itemsets):
                two_itemsets[frozenset(items)] += 1

    return two_itemsets

def calculate_association_rules(one_itemsets: dict, two_itemsets: dict, N: int) -> list:
    # timestamp = datetime.now(), если хотим добавить временную метку,
    # чтобы более старые транзакции имели меньший вес, чем более новые
    rules = []

    for source, source_freq in one_itemsets.items():
        for key, group_freq in two_itemsets.items():
            if source.issubset(key):
                target = key.difference(source)
                support = group_freq / N
                confidence = group_freq / source_freq
                # rules.append((timestamp, next(iter(source)), next(iter(target)), confidence, support))
                rules.append((next(iter(source)), next(iter(target)), confidence, support))
    
    return rules


def get_rec_list(trans_dict: dict, user, rules: list): 

    recs_list = []

    for rule in rules:
        for element_id in trans_dict[user]:
            if rule[0] == element_id:
                recs_list.append(rule)

    recs_list = sorted(recs_list, key = lambda l : l[2], reverse = True)
    
    # удаляем дубликаты, пересчитывая поддержку как среднее арифметическое
    # поддержок всех рекомендаций соответствующего элемента
    recs_dict = {}

    for rec in recs_list:
        if rec[1] not in recs_dict:
            recs_dict[rec[1]] = [rec[2], 1]
        else:
            (recs_dict[rec[1]])[0] += rec[2]
            (recs_dict[rec[1]])[1] += 1
    # для каждой рекомендации первое значение в списке - сумма уверенностей 
    # по всем появлениям рекомендации в списке, второе - число этих появлений

    recs_list_final = []
        
    for key, item in recs_dict.items():
        recs_list_final.append((key, item[0]/item[1]))
    
    # сортируем рекомендации по уровню уверенности 
    recs_list_final = sorted(recs_list_final, key = lambda t : t[1], reverse = True)

    return [t[0] for t in recs_list_final]


def main(user):
    DATA_PATH = '../data/' #путь до папки с файлами

    # формируем dataframe из csv-файла
    # если 'transactions.csv' находится в той же папке, 
    # можно просто pd.read_csv('transactions.csv', ...)
    transactions_main = pd.read_csv(
        os.path.join(DATA_PATH, 'transactions.csv'),
        dtype={
            'element_uid': np.uint16,
            'user_uid': np.uint32,
            'consumption_mode': 'category',
            'ts': np.float64,
            'watched_time': np.uint64,
            'device_type': np.uint8,
            'device_manufacturer': np.uint8
        }
    )

    # запускаем на наших данных (первые 30000 строк, чтобы время ожидания было разумным)
    transactions = transactions_main.iloc[:30000]
    transactions_dict = generate_transactions(transactions)
    one_itemsets = calculate_itemsets_one(transactions_dict)
    two_itemsets = calculate_itemsets_two(transactions_dict, one_itemsets)
    rules = calculate_association_rules(one_itemsets, two_itemsets, len(transactions_dict))

    recs_list = get_rec_list(transactions_dict, user, rules)

    # рекомендации для группы пользователей можно будет записывать в файл 
    # вместо вывода на экран
    print(json.dumps({user: recs_list[:10]}))
    
#main(240316) # пример составления ассоциативных правил для пользователя 240316