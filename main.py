import json
import pickle

import fire
import numpy as np
from collections import defaultdict

from sklearn.cluster import KMeans
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm.classes import SVC, LinearSVC

FACE_FEATURE_VECTOR_LEN = 128
USE_UNIFORM_CLUSTER_SIZE = True
K = 3
MIN_K = 2
MAX_K = 4
N_FEATURES = 13


def extract_face_features(dataset, output_path=None):
    if isinstance(dataset, str):
        print('loading dataset from {}... '.format(dataset), end='')
        with open(dataset, 'r') as f:
            dataset = json.loads(''.join(f.readlines()))
        print('done.')
    elif isinstance(dataset, list):
        dataset = dataset
    else:
        raise TypeError('dataset must be a path to a JSON file or a Numpy ndarray.')

    n_faces = len(dataset)
    print('found {} faces.'.format(n_faces))

    faces_mat = np.zeros((n_faces, FACE_FEATURE_VECTOR_LEN))

    print('extracting all faces... ', end='')
    for i, item in enumerate(dataset):
        faces_mat[i] = item['sig']
    print('done.')

    if output_path is not None:
        print('saving all extracted faces to: {}... '.format(output_path), end='')
        np.save(output_path, faces_mat)
        print('done.')

    return faces_mat


def cluster_faces(dataset, n_clusters, output_file_path=None):
    if isinstance(dataset, str):
        print('loading faces matrix...', end='')
        faces_mat = np.load(dataset)
        print('done.')
    elif isinstance(dataset, np.ndarray):
        faces_mat = dataset
    else:
        raise TypeError('dataset must be a path to a JSON file or a Numpy ndarray.')

    kmeans = KMeans(n_clusters=n_clusters, n_jobs=1, n_init=10, verbose=0)

    print('computing K-Means clustering... ', end='')
    kmeans.fit(faces_mat)
    print('done.')

    if output_file_path is not None:
        print('saving K-Means clustering object to: {}... '.format(output_file_path), end='')
        with open(output_file_path, 'wb') as f:
            pickle.dump(kmeans, f)
        print('done.')

    return kmeans


def split_by_album(dataset_path, output_file_path):
    print('loading dataset from {}... '.format(dataset_path), end='')
    with open(dataset_path, 'r') as f:
        dataset = json.loads(''.join(f.readlines()))
    print('done.')

    n_faces = len(dataset)
    print('found {} faces.'.format(n_faces))

    albums_dict = defaultdict(list)

    print('splitting items by albums... ', end='')
    for face in dataset:
        album_id = face['u_id']
        albums_dict[album_id].append(face)
    print('done.')

    print('saving results to: {}... '.format(output_file_path), end='')
    with open(output_file_path, 'w') as f:
        json.dump(albums_dict, f)
    print('done.')


def extract_features(dataset_path, output_file_path, n_clusters=10):
    print('loading dataset from {}... '.format(dataset_path), end='')
    with open(dataset_path, 'r') as f:
        dataset = json.loads(''.join(f.readlines()))
    print('done.')

    total_faces = sum(len(album) for album in dataset)
    face_feature_dict = {}
    face_feature_vectors = np.zeros((total_faces, N_FEATURES))

    for album in dataset:
        # 1. run K-Means on the entire album.
        faces_mat = extract_face_features(album)

        if USE_UNIFORM_CLUSTER_SIZE:
            kmeans = cluster_faces(faces_mat, n_clusters)

        else:
            kmeans_list = []
            for k in range(MIN_K, MAX_K + 1):
                kmeans_list.append(cluster_faces(faces_mat, k))
            inertias = [k.inertia_ for k in kmeans_list]
            derivative_1 = [inertias[i+1] - inertias[i] for i in range(0, len(inertias) - 1)]
            derivative_2 = [derivative_1[i+1] - derivative_1[i] for i in range(0, len(derivative_1) - 1)]

            for i in range(len(derivative_2)):
                if derivative_2[i] < 0:
                    break
            kmeans = kmeans_list[i+1]

        # 2. convert all faces (in the album) feature vectors to person indices.
        for face in album:
            sig = np.reshape(face['sig'], (1, -1))
            person_id = kmeans.predict(sig)
            face['person_id'] = person_id

            cluser_center = kmeans.cluster_centers_[person_id]
            person_id_conf = (((cluser_center - sig) ** 2).sum()) ** 1 / 2
            face['person_id_conf'] = person_id_conf

        # 3. calculate the percentage of appearances of each person.
        n_faces = len(album)
        appearance_perc_dict = defaultdict(lambda: 0)
        for face in album:
            person_id = int(face['person_id'])
            appearance_perc_dict[person_id] += 1 / n_faces

        sorted_appearance_perc = sorted(appearance_perc_dict.items(), key=lambda k: k[1])

        # 4. calculate the maximal number of likes for the album.
        maximum_likes = max(face['likes'] for face in album)

        # 5. calculate, for each person, the sum of his likes.

        for face in album:
            # F1: the number of likes / the maximum number of likes in the album.
            f1 = create_likes_feature(maximum_likes, face)

            # F2: if the word 'selfie' appears in the hash-tags set.
            # TODO: calculate this
            f2 = find_self(face)

            # F3: see 3.
            person_id = int(face['person_id'])
            f3 = appearance_perc_dict[person_id]

            relative_tlrb = calc_relative_tlrb(face)
            # F4: relative TL X position of the face in the photo.
            # F5: relative TL Y position of the face in the photo.
            # F6: relative BR X position of the face in the photo.
            # F7: relative BR Y position of the face in the photo.
            f4, f5, f6, f7 = relative_tlrb

            # F8: the distance of the face center from the photo center.
            f8 = calc_distance_from_center(relative_tlrb)

            # F9: the size of the face / the size of the photo.
            f9 = calc_relative_face_size(relative_tlrb)

            # F10: the confidence of the person identity
            f10 = face['person_id_conf']

            # F11-F13: hot-one vector of the person id (sorted by his appearance percentage in the album)
            f11 = 1 if face['person_id'] == sorted_appearance_perc[0][0] else 0
            f12 = 1 if face['person_id'] == sorted_appearance_perc[1][0] else 0
            f13 = 1 if face['person_id'] == sorted_appearance_perc[2][0] else 0

            face_feature_list = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13]
            face_feature_vec = np.array(face_feature_list)

            record_id = face['rec_id']
            face_feature_dict[record_id] = face_feature_vec

    for i, rec_id in enumerate(sorted(face_feature_dict.keys())):
        face_feature_vectors[i] = face_feature_dict[rec_id]

    # 6. save the feature vectors to the output file
    np.save(output_file_path, face_feature_vectors)


def create_likes_feature(maximum_likes, face):
    face_number_of_likes = face.get('likes')
    likes_percentage = face_number_of_likes / maximum_likes
    return likes_percentage


def calc_relative_tlrb(face):
    tlrb = face['tlrb']
    hw = face['hw']
    relative = (tlrb[0] / hw[0], tlrb[1] / hw[1], tlrb[2] / hw[1], tlrb[3] / hw[0])
    return relative


def calc_distance_from_center(relative_tlrb):
    y_center = (relative_tlrb[3] + relative_tlrb[0]) / 2
    x_center = (relative_tlrb[2] + relative_tlrb[1]) / 2
    d_from_center = ((y_center - 0.5) ** 2 + (x_center - 0.5) ** 2) ** 0.5
    return d_from_center


def calc_relative_face_size(relative_tlrb):
    y_size = (relative_tlrb[3] - relative_tlrb[0])
    x_size = (relative_tlrb[2] - relative_tlrb[1])
    return y_size * x_size


def find_self(face):
    hashtag = face['tags']
    hashtag_str = ''.join(hashtag).lower()
    return 1 if "self" in hashtag_str else 0


def train_classifier(x_train_path, y_train_path, x_val_path, y_val_path, type='svm', output_file_path=None):
    x_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    x_val = np.load(x_val_path)
    y_val = np.load(y_val_path)

    if type == 'svm':
        clf = LinearSVC()
    elif type == 'rf':
        clf = RandomForestClassifier(n_estimators=2)
    elif type == 'knn':
        clf = KNeighborsClassifier(n_neighbors=10)
    elif type == 'ann':
        from keras.callbacks import ReduceLROnPlateau
        from keras.layers.core import Dense
        from keras.layers.normalization import BatchNormalization
        from keras.models import Sequential
        from keras.optimizers import Adam

        model = Sequential()
        model.add(Dense(10, input_dim=x_train.shape[1], activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(10, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(5, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        clf = model

    if type != 'ann':
        clf.fit(x_train, y_train)
        print('Train score: ', clf.score(x_train, y_train))
        print('Validation score: ', clf.score(x_val, y_val))
    else:
        clf.fit(
            x=x_train,
            y=y_train,
            batch_size=128,
            epochs=100,
            verbose=2,
            callbacks=[ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)],
            validation_data=[x_val, y_val],
        )

    if output_file_path is not None:
        print('saving trained classifier to: {}'.format(output_file_path))
        with open(output_file_path, 'wb') as f:
            pickle.dump(clf, f)


def extract_all_train_features():
    input_paths = {
        'orcam/train.json': 'orcam/train_features.npy',
        'orcam/val.json': 'orcam/val_features.npy',
        'orcam/train_val.json': 'orcam/train_val_features.npy',
    }

    for input_path, output_path in input_paths.items():
        extract_features(input_path, output_path, K)


def extract_all_data_features():
    input_paths = {
        'orcam/train.json': 'orcam/train_features.npy',
        'orcam/val.json': 'orcam/val_features.npy',
        'orcam/train_val.json': 'orcam/train_val_features.npy',
    }

    for input_path, output_path in input_paths.items():
        extract_features(input_path, output_path, K)


def save_predictions(clf_file_path, features_file_path, output_file):
    with open(clf_file_path, 'rb') as f:
        clf = pickle.load(f)

    x = np.load(features_file_path)
    pred = clf.predict(x)
    conf = clf.decision_function(x)
    pred[np.abs(conf) < 1.57] = 0

    pred_str = ''.join([str(i) for i in pred])

    with open(output_file, 'w') as f:
        f.write(pred_str)


def run():
    extract_all_train_features()
    train_classifier(
        'orcam/train_features.npy',
        'orcam/train_labels.npy',
        'orcam/val_features.npy',
        'orcam/val_labels.npy',
        output_file_path='clf.pkl',
        type='svm'
    )


if __name__ == '__main__':
    fire.Fire()
