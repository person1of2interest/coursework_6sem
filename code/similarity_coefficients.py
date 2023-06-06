#коэффициент корреляции Пирсона
#вычисление сходства пользователей
def pearson(users, this_user, that_user):
    if this_usr in users and that_user in users:
        this_sum = sum(users[this_user].values())
        this_len = len(users[this_user].values())
        this_user_avg = this_sum/this_len
        
        that_sum = sum(users[that_user].values())
        that_len = len(users[that_user].values())
        that_user_avg = that_sum/that_len 
        #считаем оценку, которую пользователь ставит в среднем
        #важно, так как пользователь может быть склонен к завышению/занижению оценок
        
        this_keys = set(users[this_user].keys())
        that_keys = set(users[that_user].keys())
        #находим фильмы, которые оценили оба пользователя
        all_movies = (this_keys & that_keys) 
        
        dividend = 0
        divisor_a = 0
        divisor_b = 0
    
    for movie in all_movies:
        nr_a = users[this_user][movie] - this_user_avg #нормализуем оценки, вычитая среднюю
        nr_b = users[that_user][movie] - that_user_avg
        dividend += (nr_a) * (nr_b)
        divisor_a += pow(nr_a, 2)
        divisor_b += pow(nr_b, 2)
    
    divisor = Decimal(sqrt(divisor_a) * sqrt(divisor_b))
    
    if divisor != 0:
        return dividend/divisor
    
    return 0

#мера сходства Жаккара двух пользователей
def jaccard(users, this_user, that_user):
    if this_user in users and that_user in users:
        intersect = set(users[this_user].keys()) & set(users[that_user].keys())
        union = set(users[this_user].keys()) | set(users[that_user].keys())
        
        return len(intersect)/Decimal(len(union))
    else:
        return 0