# LACACHE: Ladder-Shaped KV Caching for Efficient Long-Range Generation of Large Language Models

## Introduction

**LaCache** is a new **KV cache optimization paradigm** designed for efficient and accurate generative inference of large language models (LLMs). It is a **training-free method** that addresses three critical requirements for long-range LLMs:

1. **Robust Long-Range Capabilities**: Enhances the ability to capture dependencies across extended contexts.
2. **Continuous Generation**: Maintains generation fluidity without interruptions.
3. **Efficient Cache Utilization**: Optimizes the use of limited memory resources.

### Key Innovations

1. **Ladder-Shaped KV Cache Pattern**:
   - Stores KV pairs both sequentially (left-to-right within each layer) and across layers (from shallow to deep).
   - Extends the span for capturing long-range dependencies under a fixed storage budget.
   - Significantly boosts long-range generation capabilities.

2. **Iterative Compaction Mechanism**:
   - Progressively compresses older caches to free up space for new tokens.
   - Ensures efficient cache utilization within a fixed cache size.

## How to Reproduce

To reproduce the results, simply run:

```bash
python pred.py
```

## Summary

LaCache provides a robust, efficient, and scalable solution for enhancing the generative inference capabilities of LLMs, enabling improved performance across long-range contexts without the need for additional training.
