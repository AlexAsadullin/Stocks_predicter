# для работы с массивами
import numpy as np
# для визуализации
import matplotlib.pyplot as plt
# для работы с данными
import pandas as pd
# для работы с датами
import datetime as dt
# для скачивания данных о цене акций
import yfinance as yf
# для масштабирования данных
from sklearn.preprocessing import MinMaxScaler
# для вычисления метрики - средне квадратичное отклонение
from sklearn.metrics import mean_squared_error  # to install type 'scikit-learn'
# блокировка вывода предупреждений от тензорфлоу
import os  # built-in no install
# модель машинного обучения и ее настройки
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# константы
COMPANY = 'AAPL'  # ticker of stock
PREDICTION_DAYS = 70  # days looking back to


def load_data():
    # загружаем данные с сайта yahoo

    # преобразуем дату к строковому формату
    start = dt.datetime(2012, 1, 1).strftime('%Y-%m-%d')
    end = dt.datetime(2020, 1, 1).strftime('%Y-%m-%d')
    # получаем данные с сайта yahoo
    data = yf.download(COMPANY, start=start, end=end, progress=False)
    # создаем синтетические признаки для улучшения метрики
    data['High^2'] = data['High'].apply(lambda x: x ** 2)
    data['Low^2'] = data['Low'].apply(lambda x: x ** 2)
    # проверяем полученные данные - смотрим первые 5 строк
    print(data.head())
    # возвращаем полученные данные
    return data


def prepare_data(data):
    # подготовка данных для обучения модели

    # масштабирвоание данных в пределах [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # создаем выборки признаков, на основе которых будем предсказывать (х), и цели, которую будем предсказывать (у)
    x_train = []
    y_train = []
    # наполняем выборки признаков и цели
    for x in range(PREDICTION_DAYS, len(scaled_data)):
        x_train.append(scaled_data[x - PREDICTION_DAYS: x, 0])
        y_train.append(scaled_data[x, 0])
    # преобразуем их к виду numpy-массива
    x_train, y_train = np.array(x_train), np.array(y_train)
    # подгоняем размерность признаков под нужный параметр
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # возвращаем выборки и модель масштабирования
    return x_train, y_train, scaler


def building_model(features):
    # создаем модель

    # получаем признаки и цели из аргументов
    x_train, y_train = features
    # инициализируем модель
    model = Sequential()
    # создаем слои нейросети для обучения модели
    model.add(LSTM(units=40, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=40, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=40))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    # назначаем функцию оптимизации и функцию ошибки
    model.compile(optimizer='adam', loss='mean_squared_error')
    # обучаем модель
    model.fit(x_train, y_train, epochs=25, batch_size=32)
    # возвращаем обученную модель
    return model


def testing_model(model, data, scaler):
    # тестируем модель

    # получаем тестовые данные с сайта yahoo
    test_start = dt.datetime(2020, 1, 1).strftime('%Y-%m-%d')
    test_end = dt.datetime.now().strftime('%Y-%m-%d')
    test_data = yf.download(tickers=COMPANY, start=test_start, end=test_end, progress=False)
    # фиксируем целевой признак теста, на нем будем проверять предсказание
    actual_prices = test_data['Close'].values
    # создаем датасет для подготовки тестовых данных
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)
    # возвращаем масштабированные тестовые признаки и целевой признак без масштабирования
    return model_inputs, actual_prices


def predictions_test(model, model_inputs, scaler):
    # получаем предсказание модели на тестовых данных

    # создаем тестовые признаки из масштабированной выборки
    x_test = []
    for x in range(PREDICTION_DAYS, len(model_inputs)):
        x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])
    # преобразуем их к numpy-массиву
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # получаем предсказание модели на тестовых признаках
    predicted_prices = model.predict(x_test)
    # ремасштабируем предсказание к реальным значениям с помощью обратного масштабирования
    predicted_prices = scaler.inverse_transform(predicted_prices)
    # возвращаем предсказание
    return predicted_prices


def plot_test(actual_prices, predicted_prices):
    # сравниваем на графике результат предсказания и реальные цены
    plt.plot(actual_prices, color='black', label=f"Actual {COMPANY} Price")
    plt.plot(predicted_prices, color='green', label=f"Predicted {COMPANY} Price")
    plt.title(f'{COMPANY} Share Price')
    plt.xlabel('Time')
    plt.ylabel(f'{COMPANY}, Share Price')
    plt.legend()
    plt.show()


def predict_next_day(model_inputs, model, scaler):
    real_data = [model_inputs[len(model_inputs) + 1 - PREDICTION_DAYS:len(model_inputs + 1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f'Prediction for tomorrow: {prediction}')


# основная программа
if __name__ == '__main__':
    # получаем сырые данные для обучения
    data = load_data()
    # получаем подготовленный датасет для обучения модели
    features = prepare_data(data=data)
    # получаем модель масштабирования
    scaler = features[-1]
    # создаем модель для обучения
    model = building_model(features=features[:2])
    # получаем подготовленные тестовые данные и реальный прайс для проверки предсказания
    model_inputs, actual_prices = testing_model(model=model, data=data, scaler=scaler)
    # делаем предсказание на тестовых данных
    predicted_prices = predictions_test(model=model, model_inputs=model_inputs, scaler=scaler)
    print(f'MSE = {mean_squared_error(predicted_prices, actual_prices)}')
    # выводим в консоль цену на следующий деньК
    predict_next_day(model_inputs=model_inputs, model=model, scaler=scaler)
    # выводим на график линии предсказанной цены и реальной цены
    plot_test(actual_prices=actual_prices, predicted_prices=predicted_prices)
    # печатаем получившуюся на тесте ошибку
