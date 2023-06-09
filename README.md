Этот репозиторий посвящён курсовой работе по рекомендательным системам, которую я выполняла в 6-м семестре (2023-й год).

Папка **data** содержит данные с конкурса [Rekko Challenge](https://boosters.pro/championship/rekko_challenge/overview), проведённом компанией Okko на платформе Boosters.pro в 2019-м году. Из всех датасетов мною использовались файлы с данными о транзакциях и оценках пользователями фильмов.

В папке **notebooks** находятся jupyter notebooks с имплементацией различных алгоритмов для рекомендательных систем:
- вариант реализации K-means и подсчёта коэффициентов подобия
- алгоритм на основе ассоциативных правил (используются данные о транзакциях) и его анализ (охват, частота попадания фильмов в рекомендации, время работы)
- алгоритм коллаборативной фильтрации в окрестности (используются данные об оценках)
- алгоритмы матричной факторизации SVD и ALS и составление рекомендаций с их помощью
- анализ алгоритмов с подбором оптимальных параметров

В папке **code** содержатся некоторые из представленных в notebooks реализаций в формате .py для удобства.

Подробнее про работу можно прочитать в отчёте (файл **ReportMaisterA6sem.pdf**)
