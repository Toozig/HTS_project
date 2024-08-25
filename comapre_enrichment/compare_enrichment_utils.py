import multiprocessing as mp
from collections import Counter
from itertools import product
import numpy as np
from scipy.spatial.distance import cosine
from functools import partial
import argparse
import os


    # Generate all possible 8-mers
ALL8_MERS = [''.join(p) for p in product('ACGT', repeat=8)]





def calculate_8mer_enrichment(sequences):

    # Count occurrences of each 8-mer
    kmer_counts = Counter()
    for seq in sequences:
        kmer_counts.update(seq[i:i+8] for i in range(len(seq) - 7))
    
    # Calculate enrichment
    total_kmers = sum(kmer_counts.values())
    enrichment = {kmer: kmer_counts.get(kmer, 0) / total_kmers for kmer in ALL8_MERS}
    
    return enrichment

def calculate_similarity(enrichment_a, sample_name, sample_seqs):
    enrichment_b = calculate_8mer_enrichment(sample_seqs)
    vector_a = np.array([enrichment_a[kmer] for kmer in sorted(enrichment_a.keys())])
    vector_b = np.array([enrichment_b[kmer] for kmer in sorted(enrichment_b.keys())])
    similarity = 1 - cosine(vector_a, vector_b)
    return sample_name, similarity

def rate_samples_by_similarity(sample_a, samples):
    # Calculate enrichment for sample_a
    enrichment_a = calculate_8mer_enrichment(sample_a)
    
    # Use multiprocessing to calculate similarities
    with mp.Pool(processes=mp.cpu_count()) as pool:
        similarity_func = partial(calculate_similarity, enrichment_a)
        similarities = pool.starmap(similarity_func, samples.items())
    
    # Sort samples by similarity (descending order)
    sorted_samples = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    return sorted_samples


def read_sequences_from_file(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        for line in file:
            seq, _ = line.strip().split(',')
            sequences.append(seq)
    return sequences

