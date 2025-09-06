# K-Nearest-Neighbours-K-NN

K-Nearest Neighbours (K-NN) is a supervised learning algorithm that predicts outcomes based on the majority class or average of the closest data points.

## 📖 Project Overview
This project is a **C++ implementation of K-Nearest Neighbour Search** optimized for high-dimensional data using tree-based indexing:  
- **KD-Tree** (median split rule).  
- **Random Projection Tree (RP-Tree)** (randomized direction split).  

The framework was tested on the **Fashion-MNIST dataset**, achieving:  
- **68% faster query performance** over linear search.  
- **15% additional performance gain** from branch pruning, without sacrificing accuracy.

## 📂 Files
- `DataVector.h / DataVector.cpp` → Implements `DataVector` class (arithmetic, norm, distance).  
- `VectorDataset.h` → Manages collections of vectors, dataset I/O.  
- `TreeIndex.h / TreeIndex.cpp` → Implements KD-Tree & RP-Tree, with singleton pattern and KNN search.  

## 🚀 How to Run
1. Compile:
   ```bash
   g++ DataVector.cpp TreeIndex.cpp -o knn -O2
