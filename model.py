import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv('your_path_to_dataset')

# Создание копии и преобразование данных
def recreate(new_data):
    new_data.iloc[:,-3:-1] = new_data.iloc[:,-3:-1].replace({'да':1,'нет':0})
    new_data = new_data.drop(new_data.columns[0:2],axis=1)
    return new_data
new_data = data.copy()
new_data = recreate(new_data)

# Нормализация данных
scaler = StandardScaler()
scaled_data = scaler.fit_transform(new_data)
normalized_data = pd.DataFrame(scaled_data, columns=new_data.columns)


def find_optimal_clusters(data, max_clusters):
    # Создание списка для сохранения инерции (суммы квадратов расстояний)
    inertias = []
    
    # Перебор различных значений количества кластеров
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        
        # Сохранение суммы квадратов расстояний в инерции
        inertias.append(kmeans.inertia_)
    
   
    
    # Автоматический выбор оптимального количества кластеров
    diff = np.diff(inertias)
    diff_r = diff[1:] / diff[:-1]
    optimal_clusters = np.argmax(diff_r) + 2
    
    return optimal_clusters

# Пример использования  
max_clusters = int(len(data)/7)  # Максимальное количество кластеров для перебора

optimal_clusters = find_optimal_clusters(normalized_data, max_clusters)
print("Оптимальное количество кластеров:", optimal_clusters)

n_clusters = optimal_clusters

def cluster_data(data, normalized_data, n_clusters):
    # Создание объекта KMeans с оптимальным количеством кластеров
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    
    # Обучение модели на данных
    kmeans.fit(normalized_data)
    
    # Получение меток кластеров для каждого объекта
    cluster_labels = kmeans.labels_
    
    # Добавление колонки с номером кластера к исходной таблице данных
    data['Кластер'] = cluster_labels
    
    return data
data = cluster_data(data, normalized_data, n_clusters)

# Функция, возвращающая один кластер
def create_database_by_value(data, column_name, value):
    # Фильтрация объектов по значению столбца
    filtered_data = data[data[column_name] == value].copy()
    
    # Создание новой базы данных с отфильтрованными объектами
    database = pd.DataFrame(filtered_data)
    
    return database