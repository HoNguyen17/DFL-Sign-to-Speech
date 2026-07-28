# DFL Sign-to-Speech

> **Decentralized Federated Learning for American Sign Language Recognition**
>
> VGU · Year 3 · Semester 2 · Distributed Systems Project

A privacy-preserving distributed machine learning system where 5 autonomous Docker nodes collaboratively train a CNN model to recognize ASL hand signs — without ever sharing raw image data. Recognized signs are converted to speech via text-to-speech output.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [System Flow](#system-flow)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Team & Responsibilities](#team--responsibilities)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Data Pipeline](#data-pipeline)
- [Model Details](#model-details)
- [gRPC Protocol](#grpc-protocol)
- [Current Status](#current-status)

---

## Overview

Traditional machine learning requires centralizing all training data on a single server — a privacy risk when the data is sensitive. This project implements **Decentralized Federated Learning (DFL)**: each of 5 nodes trains on its own private image shard, then periodically exchanges only model *weights* (not images) with peers via a gossip protocol.

**Key properties:**

- **No raw data sharing** — images stay on the node that owns them
- **Non-IID data** — each node sees a different subset of ASL classes (realistic distribution)
- **Fault-tolerant** — nodes gossip asynchronously; one slow or missing peer does not block others
- **Horizontally scalable** — adding more nodes requires no architectural changes


**To run the GUI:** Open Docker Desktop (5 nodes must be running), then run `streamlit run app_tester.py` in your terminal.
**To know how to use the GUI, please watch the demo.
---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network (dfl-net)                  │
│                                                             │
│   ┌────────┐     gRPC      ┌────────┐     gRPC      ┌────────┐
│   │ node1  │◄─────────────►│ node3  │◄─────────────►│ node5  │
│   │  CNN   │               │  CNN   │               │  CNN   │
│   │ shard1 │               │ shard3 │               │ shard5 │
│   └────────┘               └────────┘               └────────┘
│       ▲  ▲                     ▲                        ▲
│       │  └─────────────────────┤                        │
│   ┌────────┐     gRPC      ┌────────┐                   │
│   │ node2  │◄─────────────►│ node4  │◄──────────────────┘
│   │  CNN   │               │  CNN   │
│   │ shard2 │               │ shard4 │
│   └────────┘               └────────┘
│                                                             │
│   Full mesh — each node can reach all 4 peers              │
└─────────────────────────────────────────────────────────────┘
```

Each node is a Python process that runs:

1. A **gRPC server** (port 50051) to receive weight gossip from peers
2. A **training loop** to fine-tune its local CNN model on its private data shard
3. A **gossip client** to push its weights to a random peer each round

---

## System Flow

```
┌──────────────────────────────────────────────────────────┐
│  Per-node Federated Learning Round                       │
│                                                          │
│  1. Load local data shard                                │
│     └── metadata.json → sample_count (n_i)               │
│                                                          │
│  2. Train locally (LOCAL_EPOCHS)                         │
│     └── Keras CNN on private ASL images                  │
│                                                          │
│  3. Serialize weights                                     │
│     └── flatten all layers → np.float32 → bytes         │
│                                                          │
│  4. Gossip to random peer                                │
│     └── gRPC GossipWeights RPC → WeightRequest           │
│                                                          │
│  5. Receive weights from peers                           │
│     └── buffered by gRPC server in background            │
│                                                          │
│  6. FedAvg aggregation                                   │
│     └── w_global = Σ(n_i × w_i) / Σ(n_i)               │
│                                                          │
│  7. Apply aggregated weights to local model              │
│                                                          │
│  8. Sleep GOSSIP_INTERVAL → repeat                       │
└──────────────────────────────────────────────────────────┘
```

### FedAvg Formula

Weighted average by number of training samples, following McMahan et al. (2017):

```
w_global = ( n_1×w_1 + n_2×w_2 + ... + n_k×w_k ) / ( n_1 + n_2 + ... + n_k )
```

where `n_i` is the sample count on node `i` and `w_i` is its serialized weight vector.

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.10 |
| ML Framework | TensorFlow / Keras | ≥ 2.12 (CPU) |
| Transfer Learning | MobileNetV2 (ImageNet) | — |
| Communication | gRPC + Protocol Buffers | grpcio |
| Serialization | NumPy float32 bytes | — |
| Containerization | Docker + Docker Compose | — |
| Image Processing | Pillow | — |
| TTS | gTTS / pyttsx3 | — |
| UI | Streamlit | — |

---

## Project Structure

```
dfl-sign-to-speech/
│
├── app/                            # Runtime: gRPC server + FL loop
│   ├── node.py                     # Entry point — gRPC server & gossip client
│   ├── model.py                    # Model abstraction (build/serialize/deserialize)
│   ├── utils.py                    # TTS integration
│   ├── dfl_service_pb2.py          # Auto-generated proto message classes
│   └── dfl_service_pb2_grpc.py     # Auto-generated gRPC service stubs
│
├── DistributedSystemProject/       # ML models & standalone training scripts
│   ├── model_mnist.py              # CNN for 28×28 grayscale MNIST (24 classes)
│   ├── model_image.py              # CNN for 128×128 RGB images (36 classes)
│   ├── train_mnist.py              # Standalone MNIST training
│   ├── train_image.py              # Standalone image model training
│   └── utils.py                    # Weight extraction utilities
│
├── protos/
│   └── dfl_service.proto           # gRPC service contract
│
├── data_shards/                    # Partitioned training data (git-ignored)
│   ├── node1/
│   │   ├── metadata.json           # { node_id, sample_count, labels: {...} }
│   │   └── [class]/[images]
│   └── node2/ ... node5/
│
├── checkpoints/                    # Per-node model checkpoints (git-ignored)
│   └── node{1-5}/
│       ├── weights.npy
│       └── checkpoint.json
│
├── exported_models/                # Final exported models (git-ignored)
├── global_val/                     # Shared validation set (read-only in containers)
├── resized_dataset/                # Preprocessed 160×160 images
│
├── partition.py                    # Non-IID data sharding across 5 nodes
├── resize.py                       # Image preprocessing (→ 160×160 RGB)
├── auto_crop.py                    # Hand region auto-cropping utility
├── app_tester.py                   # Standalone inference & TTS demo
├── docker-compose.yml              # 5-node cluster definition
├── Dockerfile                      # Python 3.10-slim image
└── requirements.txt                # Python dependencies
```

---

## Team & Responsibilities

| Member | Role | Files | Status |
|---|---|---|---|
| **Nhan** | P2P & Orchestration | `app/node.py`, `protos/` | ✅ Functional |
| **Danh** | Model & Training | `DistributedSystemProject/model_*.py`, `train_*.py` | ✅ Done |
| **Nguyen** | Data Pipeline | `partition.py`, `resize.py`, `auto_crop.py`, `data_shards/` | ✅ Done |
| **Khang** | Docker & Networking | `docker-compose.yml`, `Dockerfile` | ✅ Done |
| **Phuc** | UI & TTS | `app/utils.py`, `app_tester.py`, Streamlit UI | ✅ Done |

---

## Getting Started

### Prerequisites

- Docker Desktop (with Compose v2)
- Python 3.10+ (for local dev / standalone training only)
- ~2 GB disk space for dataset

### 1. Clone the repository

```bash
git clone <repo-url>
cd dfl-sign-to-speech
```

### 2. Prepare the dataset

Place raw ASL images organized by class folder under `raw_dataset/`:

```
raw_dataset/
├── A/  ├── img1.jpg  └── ...
├── B/  └── ...
...
```

Then preprocess and partition:

```bash
# (Optional) Auto-crop hands from raw images
python auto_crop.py

# Resize all images to 160×160 RGB
python resize.py

# Partition into 5 non-IID shards
python partition.py
```

This creates `data_shards/node1` through `data_shards/node5`, each with a `metadata.json`.

### 3. Launch the cluster

```bash
docker-compose up --build
```

All 5 nodes start in staggered order (node1 first, then +15 s each). Each resumes from its checkpoint if one exists, then begins the FL training loop.

### 4. Monitor logs

```bash
# Tail all nodes
docker-compose logs -f

# Tail a specific node
docker-compose logs -f node1
```

Expected output:

```
[Node 1] ===== FL Round 1/30 =====
[Node 1] Training 5 epoch(s) on 42 samples...
[Node 1] Local train done. loss=0.8234  acc=78.45%
[Node 1] Round 1: Gossiped to node2 (9 MB).
[Node 1] Round 1: FedAvg applied (2 peers).
[Node 1] Checkpoint saved (round 1).
```

### 5. Stop the cluster

```bash
docker-compose down
```

### Standalone model training (optional)

```bash
pip install -r requirements.txt

python DistributedSystemProject/train_mnist.py
# or
python DistributedSystemProject/train_image.py
```

### Run the inference & TTS demo

```bash
python app_tester.py
```

### Regenerate gRPC stubs (after proto changes)

```bash
python -m grpc_tools.protoc \
  -I protos \
  --python_out=app \
  --grpc_python_out=app \
  protos/dfl_service.proto
```

---

## Configuration

All per-node settings are injected via environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|---|---|---|
| `NODE_ID` | `1` | Unique identifier for this node (1–5) |
| `NEIGHBORS` | `""` | Comma-separated hostnames of peer nodes |
| `LOCAL_EPOCHS` | `5` | Epochs to train locally per FL round |
| `GOSSIP_INTERVAL` | `10` | Seconds to sleep between rounds |
| `MAX_ROUNDS` | `30` | Total FL rounds to run |
| `NUM_CLASSES` | `26` | Number of ASL sign classes |
| `FINE_TUNE_ROUND` | `15` | Round to unfreeze backbone layers for fine-tuning |
| `BATCH_SIZE` | `8` | Training batch size per node |
| `GLOBAL_VAL_DIR` | `""` | Path to shared validation set (read-only) |

**Two-phase training:** Rounds 1–14 train only the classification head (frozen MobileNetV2 backbone). From round `FINE_TUNE_ROUND` (default 15), the top 30 backbone layers are unfrozen for full fine-tuning.

---

## Data Pipeline

### Non-IID Sharding (`partition.py`)

Real-world federated settings have non-identical data distributions across clients. This script simulates that:

1. Load all class folders from `resized_dataset/`
2. For each class: split images into 10 random shards
3. Shuffle all shards globally
4. Assign 6 shards to each of the 5 nodes (round-robin)
5. Write `metadata.json` per node

Example `data_shards/node1/metadata.json`:

```json
{
  "node_id": 1,
  "sample_count": 42,
  "labels": {
    "d": 7,
    "h": 7,
    "j": 14,
    "o": 7,
    "w": 7
  }
}
```

Node 1 only sees 5 of the 26 possible ASL classes — forcing the global model to learn from all classes through weight aggregation.

---

## Model Details

### Transfer Learning Architecture (`app/model.py`)

The federated model uses **MobileNetV2** pretrained on ImageNet as a feature extractor, with a custom classification head for 26 ASL letter classes.

```
Input: (160, 160, 3)
  → MobileNetV2 (ImageNet weights, frozen in phase 1)
  → GlobalAveragePooling2D
  → Dense(128, relu)
  → Dropout(0.3)
  → Dense(26, softmax)
```

### Standalone Image CNN (`DistributedSystemProject/model_image.py`)

Trained from scratch on 128×128 RGB images, 36 ASL signs (A–Z + 0–9):

```
Input: (128, 128, 3)
  → Conv2D(32, 3×3, relu)
  → Conv2D(64, 3×3, relu) → MaxPooling2D
  → Conv2D(128, 3×3, relu) → MaxPooling2D
  → Flatten → Dense(128, relu) → Dense(36, softmax)
```

### Standalone MNIST CNN (`DistributedSystemProject/model_mnist.py`)

Trained on 28×28 grayscale images, 24 ASL letters (A–X, excluding J and Z):

```
Input: (28, 28, 1)
  → Conv2D(32, 3×3, relu) → MaxPooling2D
  → Conv2D(64, 3×3, relu) → MaxPooling2D
  → Flatten → Dense(128, relu) → Dense(24, softmax)
```

### Weight Serialization

Weights cross the network as raw bytes — no framework-specific format, no overhead:

```python
# Serialize (sender)
flat = np.concatenate([w.flatten() for w in model.get_weights()]).astype(np.float32)
payload = flat.tobytes()   # ~9 MB per gossip message

# Deserialize (receiver)
flat = np.frombuffer(data, dtype=np.float32).copy()
# reshape layer-by-layer and call model.set_weights(...)
```

---

## gRPC Protocol

Defined in `protos/dfl_service.proto`:

```protobuf
syntax = "proto3";

service DFLService {
  rpc GossipWeights (WeightRequest) returns (WeightResponse);
}

message WeightRequest {
  int32 node_id   = 1;   // Sender node ID
  bytes model_data = 2;  // Serialized np.float32 weight vector
  int32 sample_count = 3; // Sender's training sample count (for FedAvg)
}

message WeightResponse {
  bool success = 1;
}
```

All nodes listen on port **50051**. The gRPC server runs in a background thread while the FL loop runs on the main thread. Incoming weights are buffered (up to 10 entries, ~90 MB max) and drained each round during FedAvg.

---

## Current Status

| Component | Status | Notes |
|---|---|---|
| gRPC server / gossip client | ✅ Functional | Sends and receives weight bytes |
| FedAvg aggregation | ✅ Functional | Weighted average over buffered peers |
| MobileNetV2 transfer model | ✅ Functional | Two-phase training (freeze → fine-tune) |
| Image CNN (128×128) | ✅ Functional | Trains standalone |
| MNIST CNN (28×28) | ✅ Functional | Trains standalone |
| Data sharding | ✅ Complete | 5 non-IID shards with metadata |
| Docker cluster | ✅ Functional | All 5 nodes boot and mesh correctly |
| Model checkpointing | ✅ Functional | Save/load weights between rounds |
| TTS output | ✅ Functional | gTTS + pyttsx3 via `app_tester.py` |
| Streamlit UI | ✅ Functional | Live inference demo |

---

## References

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg), 2017
- MobileNetV2: Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, 2018
- gRPC Python documentation: [grpc.io](https://grpc.io/docs/languages/python/)
- ASL MNIST dataset: [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- TensorFlow / Keras documentation: [tensorflow.org](https://www.tensorflow.org/)