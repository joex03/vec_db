import numpy as np
from typing import Dict, List, Annotated
from sklearn.cluster import KMeans
from collections import defaultdict
from linecache import getline
import os
DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70
class VecDB:

    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.dat", new_db=True, db_size=None) -> None:
     self.db_path = database_file_path
     self.index_path = index_file_path
     self.file_path = 'data'
     if new_db:
        if db_size is None:
            raise ValueError("You need to provide the size of the database")
        # delete the old DB file if exists
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.generate_database(db_size)
    
    def calculate_similarity(self, node1, node2) -> float:
        dot_product = np.dot(node1, node2)
        norm_vec1 = np.linalg.norm(node1)
        norm_vec2 = np.linalg.norm(node2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self.build_index(vectors)
        
    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)    
        
    def select_parameters(self):
        self.num_centroids = int(np.ceil(np.sqrt(self.data_size)))
            
        self.records_per_read = 1000
        
        self.clusters_uncertainty = int(self.num_centroids // 4)
        
        self.kmeans_iterations = 25
            
    def load_codebooks(self):
        self.select_parameters()
        codebooks = None
        with open(self.codebooks_file_path, "r") as fin:
            codebooks = np.loadtxt(fin, delimiter=",", dtype=np.float32, max_rows=self.num_centroids)
        self.codebooks = codebooks
    
    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        if isinstance(rows, np.ndarray) and rows.shape[1] == 70:
            convert_rows = False
        else:
            convert_rows = True
        if convert_rows:
            rows = np.array([row["embed"] for row in rows])
        self.data_size = len(rows)
        
        self.select_parameters()
                
        # Ensure that self.file_path directory exists
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)
            
        # Save the data in files each file has records_per_read records
        for i in range(0, self.data_size, self.records_per_read):
            with open(f'{self.file_path}/{i}', "w") as fout:
                np.savetxt(fout, rows[i:i+self.records_per_read], delimiter=",", fmt='%.8f')
        
        # Train only on 1 million, if data is more than 1 million, else, train on all data
        training_data = None
        if self.data_size > 1000000:
            training_data = rows[:1000000]
        else:
            training_data = rows
            
        # Train Kmeans on all vectors
        codebooks = None
        kmeans_model = KMeans(n_clusters=self.num_centroids, n_init=10, max_iter=self.kmeans_iterations, init='random')
        kmeans_model.fit(training_data)
        codebooks = kmeans_model.cluster_centers_
        print(f"Finished training model")
            
        self.codebooks = codebooks
        
        # Save the codebooks to the codebooks file
        with open(f'codebooks', "w") as fout:
            np.savetxt(fout, codebooks, delimiter=",")

        pq_codes = np.zeros((self.data_size), dtype=np.uint32)
           
        # Save the inverted index to the inverted index file
        with open(f'inverted_index', "w") as fout: 
            # Update the inverted index during insertion
            # Predict the centroid of each subvector for each record
            pq_codes = kmeans_model.predict(rows)
            for i in range(self.num_centroids):
                centroid_records = np.where(pq_codes == i)[0]
                fout.write(','.join([str(id) for id in centroid_records])+"\n")
            
            print(f"Finished predicting all records")
                
        self.inverted_index_path = 'inverted_index'
    def build_index(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        self.insert_records(rows)
                
    def retrieve(self, query: Annotated[List[float], 70], top_k = 1):     
        print(f"enter")
        query = query[0]
        
        # Caclulate eculidean distance between all the query vector and each of the centroids, and store it in distances matrix
        query_centroids_distances = np.zeros((self.num_centroids))
        query_centroids_distances = np.linalg.norm(self.codebooks - query, axis=1)
        
        # Use the inverted index to filter potential records
        potential_records = set()
        indexes = np.argsort(query_centroids_distances)[:self.clusters_uncertainty]
        for index in indexes:
            # Get the records ids from the inverted index file, using the index of the centroid, and the subvector number
            idsLine = getline(self.inverted_index_path, index + 1)[:-1]
            if idsLine == '':
                continue
            ids = [np.uint32(id) for id in idsLine.split(',')]
            potential_records.update(ids)
        
        records = np.zeros((len(potential_records), 71),dtype=np.float32)
        
        # Read only the potential records from the files
        for i,record_id in enumerate(potential_records):
            # read the line and append it to records list with linecache module
            file_num = (record_id // self.records_per_read) * self.records_per_read
            record = getline(f'{self.file_path}/{file_num}', (record_id % self.records_per_read) + 1).split(',')
            records[i] = np.array([record_id] + [np.float32(i) for i in record])
        # sort the records based on cosine similarity between each record and query vector 
        records = sorted(records, key=lambda x: self.calculate_similarity(x[1:], query), reverse=True)
        print(f"exit")
        # Return the top_k records ids
        return [np.int32(record[0]) for record in records[:top_k]]
