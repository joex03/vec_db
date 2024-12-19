import pickle
from typing import List, Annotated
import numpy as np
import os
from sklearn.cluster import KMeans, MiniBatchKMeans

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.dat", new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.index = None
        self.cluster_centers = None
        self.inverted_index = None
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
        else:
            self.load_index()

    def generate_database(self, size: int) -> None:
        print("Generating database...")
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()
        print("Database generated")

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        print("Retrieving results")
        
        # Find the nearest cluster center
        distances = np.linalg.norm(self.cluster_centers - query, axis=1)
        nearest_cluster = np.argmin(distances)
        
        # Retrieve vectors from the nearest cluster
        candidate_indices = self.inverted_index[nearest_cluster]
        candidate_vectors = np.array([self.get_one_row(idx) for idx in candidate_indices])
        
        scores = [(self._cal_score(query, vec), idx) for vec, idx in zip(candidate_vectors, candidate_indices)]
        scores = sorted(scores, reverse=True)[:top_k]
        print("Results retrieved")
        return [s[1] for s in scores]

    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        print("Building index")
        data = self.get_all_rows()
        num_records = len(data)
        n_clusters = int(np.sqrt(num_records))  
        
        # clusing using minibatch
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=DB_SEED_NUMBER, batch_size=1000, n_init=10, max_no_improvement=10, verbose=0)
        kmeans.fit(data)
        self.cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        self.inverted_index = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(labels):
            self.inverted_index[label].append(idx)
        
        # Save the index to a file
        with open(self.index_path, 'wb') as f:
            pickle.dump({'cluster_centers': self.cluster_centers, 'inverted_index': self.inverted_index}, f)
        print("Index built and saved")

    def load_index(self) -> None:
        with open(self.index_path, 'rb') as f:
            data = pickle.load(f)
        self.cluster_centers = data['cluster_centers']
        self.inverted_index = data['inverted_index']