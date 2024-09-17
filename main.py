import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from pyransac3d import Plane


def load_points_from_csv(file_name):

    #Wczytuje chmurę punktów z pliku CSV.

    df = pd.read_csv(file_name)
    points = df[['x', 'y', 'z']].to_numpy()  # Zakładamy, że CSV ma kolumny: 'x', 'y', 'z'
    return points


def find_disjoint_clusters_dbscan(points, eps=0.5, min_samples=10):

    # Znajduje rozłączne chmury punktów za pomocą algorytmu DBSCAN.
    # Stworzenie i dopasowanie modelu DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(points)

    labels = dbscan.labels_

    return labels


def save_clusters_to_csv(points, labels):

    # Zapisuje rozłączne chmury punktów do oddzielnych plików CSV.

    unique_labels = set(labels)
    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue  # Pomijamy szum (punkty oznaczone jako -1)
        cluster_points = points[labels == cluster_id]
        df = pd.DataFrame(cluster_points, columns=["x", "y", "z"])
        file_name = f"cluster_{cluster_id + 1}.csv"
        df.to_csv(file_name, index=False)
        print(f"Zapisano klaster {cluster_id + 1} do pliku {file_name}.")


def fit_plane_ransac_pyransac(points):

    #Dopasowanie płaszczyzny do chmury punktów za pomocą pyransac3d.

    plane = Plane()
    best_eq, inliers = plane.fit(points, thresh=0.01)  # Dopasowanie płaszczyzny przy użyciu pyransac3d
    return best_eq


def analyze_plane(plane):

    # Analiza, czy chmura punktów jest płaszczyzną i czy jest pionowa czy pozioma.

    a, b, c, d = plane
    normal_vector = np.array([a, b, c])
    norm = np.linalg.norm(normal_vector)

    # Sprawdzenie, czy jest płaszczyzną
    if np.isclose(norm, 0):
        print("Chmura punktów nie tworzy płaszczyzny.")
        return

    print(f"Wektor normalny do płaszczyzny: [{a:.2f}, {b:.2f}, {c:.2f}]")

    normal_vector_normalized = normal_vector / norm

    # Określenie, czy płaszczyzna jest pozioma czy pionowa
    if np.isclose(normal_vector_normalized[2], 0):
        print("Płaszczyzna jest pionowa.")
    elif np.isclose(normal_vector_normalized[0], 0) and np.isclose(normal_vector_normalized[1], 0):
        print("Płaszczyzna jest pozioma.")
    else:
        print("Płaszczyzna jest ukośna.")


def visualize_points_and_plane(points, plane, cluster_id):

    # Wizualizacja chmury punktów oraz dopasowanej płaszczyzny

    a, b, c, d = plane


    xx, yy = np.meshgrid(np.linspace(min(points[:, 0]), max(points[:, 0]), 100),
                         np.linspace(min(points[:, 1]), max(points[:, 1]), 100))
    zz = (-a * xx - b * yy - d) / c

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=f'Punkty klastra {cluster_id + 1}')
    ax.plot_surface(xx, yy, zz, color='r', alpha=0.5, label='Dopasowana płaszczyzna')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'Dopasowanie płaszczyzny do klastra {cluster_id + 1} za pomocą pyransac3d')
    plt.show()



file_name = "polaczone.csv"
points = load_points_from_csv(file_name)


labels = find_disjoint_clusters_dbscan(points, eps=0.5, min_samples=30)


save_clusters_to_csv(points, labels)


unique_labels = set(labels)
for cluster_id in unique_labels:
    if cluster_id == -1:
        continue  # Pomijamy szum (punkty oznaczone jako -1)

    cluster_points = points[labels == cluster_id]
    print(f"\nAnaliza klastra {cluster_id + 1}:")
    if len(cluster_points) < 3:
        print("Za mało punktów, aby dopasować płaszczyznę.")
        continue

    # Dopasowanie płaszczyzny za pomocą pyransac3d
    plane_parameters = fit_plane_ransac_pyransac(cluster_points)
    if plane_parameters:
        analyze_plane(plane_parameters)
        visualize_points_and_plane(cluster_points, plane_parameters, cluster_id)
    else:
        print("Nie udało się dopasować płaszczyzny.")