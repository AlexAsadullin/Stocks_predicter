'''import sqlite3  # for databases
from config import host, user, password, db_name

try:
    connection = sqlite3.connect("predict_history.sqlite")
    with connection.cursor() as cursor:
        create_table_query = (
            "INSERT INTO history (id, ticker, dates, time_of prediction, predicted_price, actual_price)"
            " VALUES(2, 'IBM', '2024-04-18', '12:00', 150.1, 120,08)")
        cursor.execute(create_table_query)
        # connection.commit()
    connection.close()
except Exception as ex:
    print('connection refused')
    print(ex)
'''

# Импорт библиотеки
import sqlite3

# Подключение к БД
con = sqlite3.connect("predict_history.sqlite")
print('db connected')
# Создание курсора
cur = con.cursor()
# Выполнение запроса и получение всех результатов
cur.execute(
    "INSERT INTO history (id, ticker, prediction_date, time_of_prediction, predicted_price, actual_price)"
    " VALUES(2, 'APPL', '2024-04-18', '12:00', 150.1, 120.08)")
cur.execute('SELECT * FROM history')
print('changes completed')
con.commit()
con.close()
