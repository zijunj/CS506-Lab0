## Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    v3 = dot_product(vector1,vector2)
    normv1 = np.linalg.norm(vector1)
    normv2 = np.linalg.norm(vector2)
    
    result = cosine_similarity(vector1,vector2)
    
    expected_result = v3 / (dot_product(normv1,normv2))
    
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    target_vector=np.array([1, 2, 3])
    vectors=np.matrix([[1,3,2],[4,5,6],[3,4,5],[2,3,4]])
    closest_vector=cosine_similarity(target_vector,vectors[0])
    index=0
    for i in vectors:
        if cosine_similarity(target_vector,vectors[i]) > closest_vector:
            closest_vector = cosine_similarity(target_vector,vectors[i])
            index = i

    result = test_nearest_neighbor(target_vector,vectors)
    
    expected_index = index
    
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
