# Advancing Protein-Protein Interaction Prediction with Graph Neural Networks and Language Models

This repository contains the official code for the project **"Advancing Protein-Protein Interaction Prediction with Graph Neural Networks and Language Models."** This project replicates and extends the PPI prediction framework introduced by Jha et al., leveraging advanced graph-based deep learning techniques and protein language models.

**[Link to the Project Report](./Bioinfromatics___project_Report.pdf)**

## 📋 Project Overview

Protein-Protein Interactions (PPIs) are fundamental to nearly all biological processes. This project focuses on predicting these interactions using computational methods to overcome the time and resource limitations of traditional experimental approaches.

We employ Graph Neural Networks (GNNs) to model proteins as graph structures, where amino acid residues are nodes and their spatial proximity defines the edges. The node features are derived from powerful pre-trained language models like **ProtBERT** and **ProstT5**. Our work replaces the SeqVec embeddings used in the original paper with ProstT5, a state-of-the-art model that integrates both protein sequence and structural information, leading to improved prediction accuracy.

### ✨ Key Features

* **Graph-Based PPI Prediction**: Utilizes Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs) to learn from protein structures.
* **Advanced Embeddings**: Integrates node features from pre-trained protein language models, including ProtBERT and ProstT5, to capture rich contextual information.
* **Robust Graph Construction**: Uses the `Graphein` library to build accurate and reliable protein graphs, addressing structural issues found in earlier tools.
* **Superior Performance**: Our implementation with ProstT5 embeddings achieves a **97% accuracy**, demonstrating the effectiveness of combining sequence and structural data for PPI prediction.
* **Reproducibility**: The repository provides detailed instructions, datasets, and pre-trained models to ensure full reproducibility of our results.

## 🛠️ Methodology

Our approach can be broken down into four main stages:

1.  **Data Curation**: We used the human PPI dataset from the original base paper. The corresponding PDB files for 3,978 unique proteins were downloaded from the RCSB Protein Data Bank.

2.  **Graph Construction**:
    * Each protein is represented as a graph where nodes are amino acid residues.
    * Edges are created between residues if their Euclidean distance is less than 5Å.
    * Node features were generated using multiple methods: one-hot encoding, physicochemical properties, **ProtBERT embeddings**, and **ProstT5 embeddings**.

3.  **Model Architecture**:
    * We designed a Siamese-style architecture with two parallel GNN branches (one for each protein in a pair).
    * Each branch uses either a GCN or a GAT to generate a graph embedding for a protein.
    * These two embeddings are concatenated and passed through a fully connected network to predict the final interaction probability.

4.  **Training and Optimization**:
    * Models were trained using the Adam optimizer and Binary Cross Entropy (BCE) loss.
    * We performed extensive hyperparameter tuning using Weights & Biases (W&B) Sweeps to find the optimal configuration for learning rate, dropout, and other key parameters.
    * To handle large graph sizes, graphs were dynamically loaded at the batch level during training.

## 📊 Results

Our experiments show that GNNs combined with language model embeddings are highly effective for PPI prediction.

* **GAT vs. GCN**: GAT models consistently outperformed GCNs across all feature types, highlighting the benefit of attention mechanisms in focusing on critical residues.
* **Embedding Performance**: Pre-trained embeddings from ProstT5 and ProtBERT significantly outperformed baseline features like one-hot encoding and physicochemical properties.
* **Best Overall Model**: The GAT model using **ProstT5 embeddings** achieved the highest performance, with an accuracy, precision, and F1-score of **97-98%**. This validates our hypothesis that embeddings integrating both sequence and 3D structure are superior for this task.

| GNN Model | Node Features | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| GCN | ProtBERT embeddings | 0.93 | 0.96 | 0.95 | 0.95 |
| GCN | **ProstT5 embeddings** | **0.97** | **0.98** | **0.97** | **0.98** |
| GAT | ProtBERT embeddings | 0.95 | 0.98 | 0.94 | 0.96 |
| GAT | **ProstT5 embeddings** | **0.97** | **0.98** | **0.97** | **0.98** |

## 🚀 Getting Started

The code is publicly available on GitHub. To get started, clone the repository and follow the instructions in the provided text file.

```bash
git clone [https://github.com/abdulkalam556/GNN-PPI.git](https://github.com/abdulkalam556/GNN-PPI.git)
cd GNN-PPI
```
A file named `commands_to_run_project.txt` is included in the repository, containing all the necessary commands to:

* Download the required PDB files.
* Generate protein embeddings.
* Construct the protein graphs.
* Train and test the GNN models.

## 👥 Team Contributions

* **Abdul Kalam Azad Shaik**: Led data collection, preprocessing, PDB file management, embedding extraction, and graph construction.
* **Ashfaq Sohail Shaik**: Implemented the GAT model architecture, designed the training pipelines, and executed the hyperparameter search.
* **Phalguna Peravali**: Validated graph correctness, implemented the GCN model, and handled the calculation and analysis of test metrics.

## 📚 References

> \[1] Jha, K., Saha, S., & Singh, H. (2022). Prediction of protein-protein interaction using graph neural networks. [cite_start]*Scientific Reports, 12*(1), 8360.
>
> \[2] Heinzinger, M., Weissenow, K., Sanchez, J., Henkel, A., Steinegger, M., & Rost, B. (2023). [cite_start]Prostt5: Bilingual language model for protein sequence and structure.
>
> \[3] Elnaggar, A., Heinzinger, M., Dallago, C., Rehawi, G., Wang, Y., Jones, L., ... & Steinegger, M. (2021). Prottrans: Toward understanding the language of life through self-supervised learning. *IEEE transactions on pattern analysis and machine intelligence, 44*(10), 7112-7127.
>
> \[4] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017). Graph attention networks. *arXiv preprint arXiv:1710.10903*.
>
> \[5] Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. *arXiv preprint arXiv:1609.02907*.
