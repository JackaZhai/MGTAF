# Multiplex Graph Transformer with Adaptive Fusion (MGTAF)

This project proposes the MGTAF model, aiming to achieve high-precision diagnosis of Autism Spectrum Disorder (ASD) through multimodal functional brain networks.

## Project Overview

The MGTAF model innovatively combines Graph Convolutional Networks (GCN) with Transformer architecture, integrates multiple functional connectivity patterns (Pearson correlation, sparse representation, Granger causality), and enhances ASD classification accuracy through an adaptive fusion mechanism. This method achieves a classification accuracy of 94.7% on the public ABIDE dataset, significantly outperforming existing methods.

## Directory Structure

```
data/                   # Data storage directory
layers/
  module.py             # Network layer modules
  transformer.py        # Transformer-related implementation
Model/
  model.py              # MGTAF model definition
util/
  TripleData.py         # Triplet data processing tools
  TripleDataset.py      # Triplet dataset definition
```

## Main Features

- **Multimodal Brain Network Modeling**: Supports Pearson correlation, sparse representation, and Granger causality connectivity patterns.
- **GCN-Transformer Hybrid Architecture**: Captures both local topological features and global neural dependencies.
- **MixPool Adaptive Pooling**: Automatically balances mean pooling and max pooling to enhance feature representation.
- **Bidirectional Interactive Attention and Adaptive Fusion**: Dynamically balances different modal features to improve model discriminability.

## Quick Start

1. Install dependencies
2. Data preparation

- Place your triplet brain network data in the `data/` directory. For format requirements, see `util/TripleData.py`.

3. Train the model

## Related Paper Information

- Dataset: ABIDE, 45 ASD and 47 controls, ages 7-15. For preprocessing and brain region segmentation, see the Materials and Methodology section of the paper.
- Main methods: Multimodal brain network construction, GCN-Transformer hybrid feature extraction, MixPool pooling, bidirectional interactive attention, DSAF adaptive fusion.
- Results: MGTAF achieves 94.7% accuracy on the ABIDE dataset, outperforming baseline methods such as MPGCN and GAT.
