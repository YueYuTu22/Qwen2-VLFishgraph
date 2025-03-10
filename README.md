# Multimodal Prediction for Fish Disease Knowledge Graph

**Undergraduate Thesis Project**  
**[Yue Yu]**  

## Abstract

In this work, we present a multimodal knowledge graph embedding framework that integrates textual and visual information for improved entity-relation modeling in fish disease prediction. Our approach combines the RotatE algorithm with Qwen2-VL, a vision-language model, to enhance knowledge graph representations by incorporating image-based features. We employ a cross-attention mechanism to effectively fuse textual and visual embeddings for a unified semantic space. A contrastive learning strategy with hard negative sampling is introduced to refine entity differentiation. The model is trained using a margin ranking loss and optimized with different learning rate schedulers. Experimental results demonstrate that our approach achieves good ranking performance. These findings highlight the potential of multimodal knowledge graphs in improving predictive accuracy for fish disease identification. Future work includes extending the framework to additional modalities and enhancing multimodal fusion techniques.

# Qwen2-VLFishgraph
Multimodal + Knowledge Graph + RotatE + Qwen2-VL

## System Configuration

| System Hardware    | Value                     |
|--------------------|--------------------------|
| CPU Count         | 8                          |
| Logical CPU Count | 16                         |
| GPU Count        | 1                          |
| GPU Type         | NVIDIA GeForce RTX 4090    |

## Training and Validation Performance Metrics

| Metric                      | Value   |
|-----------------------------|--------:|
| Epochs                      | 99      |
| Global Steps                | 79,799  |
| Learning Rate (AdamW)       | 0.00005 |
| Training Loss (Final)       | 1.3478  |
| Validation Loss             | 1.3477  |
| Training MRR                | 0.96875 |
| Validation MRR              | 1.0     |
| Training Rank-1 Accuracy    | 1.0     |
| Validation Rank-1 Accuracy  | 1.0     |
