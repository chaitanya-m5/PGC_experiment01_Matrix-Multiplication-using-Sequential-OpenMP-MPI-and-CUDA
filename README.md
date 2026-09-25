# Parallel Computing (PGC Lab) - Experiment 1: $4000 \times 4000$ Matrix Multiplication

<a name="1-project-overview"></a>
## 1. Project Overview

The objective of this laboratory experiment is to implement, evaluate, and compare four fundamental parallel computing paradigms using a dense $4000 \times 4000$ matrix multiplication workload ($C = A \times B$)[cite: 2]:

1. **Sequential Baseline:** Single-threaded execution on a single CPU core to establish benchmark execution time[cite: 2].
2. **OpenMP Shared-Memory Parallelism:** Multi-threaded parallel processing utilizing shared host RAM across multiple logical CPU cores[cite: 2].
3. **MPI Distributed Memory:** Multi-node parallel processing across separate virtual machines with independent address spaces using network message passing *(WIP)*[cite: 2].
4. **CUDA GPU Acceleration:** Massively parallel kernel execution leveraging thousands of lightweight GPU threads *(WIP)*[cite: 2].

### Mathematical Problem & Verification
Matrix multiplication is an $O(N^3)$ computational workload requiring $N^3$ floating-point multiplications and additions[cite: 2].
* **Matrix Dimensions:** $N = 4000$ ($4000 \times 4000$ matrices, totaling 16 million elements per matrix)[cite: 2].
* **Initialization:** All input elements $A[i][j] = 1.0$ and $B[i][j] = 1.0$[cite: 2].
* **Verification Check:** Every element of output matrix $C$ is computed as the dot product of a row from $A$ and a column from $B$[cite: 2]:
  $$C[0][0] = \sum_{k=0}^{3999} (A[0][k] \times B[k][0]) = \sum_{k=0}^{3999} (1.0 \times 1.0) = 4000.00$$
[cite: 2]

---

## 📌 Table of Contents
* [1. Project Overview](#1-project-overview)
* [2. System & Environment Specifications](#2-system--environment-specifications)
* [3. Part A: Sequential Matrix Multiplication](#3-part-a-sequential-matrix-multiplication)
* [4. Part B: OpenMP Shared-Memory Parallelism](#4-part-b-openmp-shared-memory-parallelism)
* [5. Part C: MPI Distributed Memory (WIP)](#5-part-c-mpi-distributed-memory-wip)
* [6. Part D: CUDA GPU Acceleration (WIP)](#6-part-d-cuda-gpu-acceleration-wip)
* [7. Results & Performance Comparison](#7-results--performance-comparison)
* [8. Key Observations & Analysis](#8-key-observations--analysis)

---

<a name="2-system--environment-specifications"></a>
## 2. System & Environment Specifications

* **Operating System:** Windows 11 Home with WSL2 Ubuntu 24.04 LTS (`chaitanya@Chaitanya`)[cite: 2, 3]
* **Compiler:** `gcc` version 13.3.0 (`Ubuntu 13.3.0-6ubuntu2~24.04.1`)[cite: 3]
* **Available CPU Threads:** 12 Logical Cores (`nproc = 12`)
* **Development Packages:** `build-essential`[cite: 2, 3]

---

<a name="3-part-a-sequential-matrix-multiplication"></a>
## 3. Part A: Sequential Matrix Multiplication

<details open>
<summary><b>Click to Expand/Collapse Part A Details</b></summary>

### Paradigm Overview
Sequential matrix multiplication executes on a single processor core in a deterministic, linear sequence[cite: 2]. It uses three nested loops ($i$, $j$, $k$) to multiply row vectors of matrix $A$ by column vectors of matrix $B$[cite: 2]. 

Because no parallelization is active, execution speed is bound by single-thread CPU clock performance and memory access latency[cite: 2]. Non-contiguous memory accesses when traversing columns of matrix $B$ ($B[k \cdot N + j]$) lead to CPU cache misses, making this the baseline reference implementation for speedup calculations[cite: 2].

### Experimental Execution
* **Working Directory:** `~/parallel_lab/sequential`[cite: 2, 3]
* **Optimization Level:** `-O2` (Enables standard loop and instruction-level optimizations)[cite: 2, 3]
* **Recorded Execution Time:** `437.471680 seconds`[cite: 3]
* **Verification Check:** `C[0][0] = 4000.00`[cite: 3]

### Execution Screenshot
![Sequential Matrix Multiplication Result](./partA-sequential-result.png)

</details>

---

<a name="4-part-b-openmp-shared-memory-parallelism"></a>
## 4. Part B: OpenMP Shared-Memory Parallelism

<details open>
<summary><b>Click to Expand/Collapse Part B Details</b></summary>

### Paradigm Overview
OpenMP (Open Multi-Processing) utilizes a fork-join threading model for shared-memory multi-core architectures[cite: 2]. Memory is shared globally across all threads, eliminating the need to move matrix data across network interfaces[cite: 2].

The `#pragma omp parallel for` compiler directive dynamically splits the outer loop ($i$-loop iterations) among available host worker threads[cite: 2]. Each thread processes a independent block of matrix rows while sharing read access to $A$ and $B$, avoiding synchronization locks during calculation[cite: 2].

### Experimental Execution
* **Working Directory:** `~/parallel_lab/openmp`[cite: 2]
* **Allocated Threads:** 12 Logical CPU Threads (`export OMP_NUM_THREADS=12`)
* **Compilation Flags:** `gcc -O2 -fopenmp`[cite: 2]
* **Recorded Execution Time:** `202.132210 seconds`
* **Verification Check:** `C[0][0] = 4000.00`

### Execution Screenshot
![OpenMP Matrix Multiplication Result](./04-partB-openmp-results.png)

</details>

---

<a name="5-part-c-mpi-distributed-memory-wip"></a>
## 5. Part C: MPI Distributed Memory (WIP)

<details>
<summary><b>Click to Expand/Collapse Part C Details</b></summary>

### Paradigm Overview
The Message Passing Interface (MPI) model targets distributed-memory systems where individual processing nodes do not share RAM[cite: 2]. Each process executes in its own isolated memory address space, communicating explicit messages across network interfaces[cite: 2].

### Architectural Workdistribution
1. **Data Scatter (`MPI_Scatter`):** Master rank divides matrix $A$ into contiguous row slices (e.g., 1000 rows per rank) and distributes them to worker nodes[cite: 2].
2. **Data Broadcast (`MPI_Bcast`):** Master broadcasts the full matrix $B$ to all participating ranks so every process can complete its inner-loop dot products[cite: 2].
3. **Data Gather (`MPI_Gather`):** Each rank computes its assigned partial rows of matrix $C$, which are gathered back to Rank 0 to reconstruct the complete $4000 \times 4000$ output[cite: 2].

*(Screenshots and benchmarks will be updated upon final cluster execution)*[cite: 2]

</details>

---

<a name="6-part-d-cuda-gpu-acceleration-wip"></a>
## 6. Part D: CUDA GPU Acceleration (WIP)

<details>
<summary><b>Click to Expand/Collapse Part D Details</b></summary>

### Paradigm Overview
CUDA (Compute Unified Device Architecture) employs Single Instruction, Multiple Threads (SIMT) GPU acceleration[cite: 2]. Unlike CPUs optimized for low-latency sequential tasks, GPUs feature thousands of small cores designed for high-throughput parallel compute blocks[cite: 2].

### Execution & Memory Workflow
1. **Host-to-Device Copy (`cudaMemcpy`):** CPU allocates VRAM on the GPU (`cudaMalloc`) and transfers matrices $A$ and $B$ across PCIe bus[cite: 2].
2. **Grid & Block Configuration:**
   * **Block Size:** $16 \times 16 = 256$ threads per block[cite: 2].
   * **Grid Size:** $(4000/16) \times (4000/16) = 250 \times 250 = 62,500$ blocks[cite: 2].
   * **Total Threads:** $62,500 \times 256 = 16,000,000$ logical CUDA threads executing concurrently[cite: 2].
3. **Device-to-Host Copy:** Computed result $C$ is copied back to CPU system memory for verification[cite: 2].

*(Screenshots and benchmarks will be updated upon GPU environment driver configuration)*[cite: 2]

</details>

---

<a name="7-results--performance-comparison"></a>
## 7. Results & Performance Comparison

All executed tests evaluate dense $4000 \times 4000$ matrix multiplication with output correctness verified at $C[0][0] = 4000.00$[cite: 2, 3].

| Computing Model | Parallel Paradigm | Hardware / Cores | Execution Time (s) | Calculated Speedup | Execution Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sequential** | Single CPU Core | 1 Core | **437.471680 s**[cite: 3] | **1.00× (Baseline)** | Completed[cite: 3] |
| **OpenMP** | Shared-Memory Parallel | 12 CPU Threads | **202.132210 s** | **2.16×** | Completed |
| **MPI** | Distributed Memory | 4 VM Cluster | *TBD*[cite: 2] | *TBD* | Pending[cite: 2] |
| **CUDA** | GPU Hardware Acceleration | NVIDIA GPU | *TBD*[cite: 2] | *TBD* | Pending[cite: 2] |

### Speedup Calculation
$$\text{Speedup} = \frac{T_{\text{sequential}}}{T_{\text{parallel}}} = \frac{437.471680\text{ s}}{202.132210\text{ s}} \approx 2.16\times$$
[cite: 3]

---

<a name="8-key-observations--analysis"></a>
## 8. Key Observations & Analysis

1. **Algorithm Correctness:** Both Sequential baseline and multi-threaded OpenMP implementations returned $C[0][0] = 4000.00$, confirming mathematical precision across execution environments[cite: 3].
2. **OpenMP Parallel Scaling:** Multi-threading across 12 logical CPU cores reduced execution latency from ~437.47 seconds to ~202.13 seconds, demonstrating a **2.16× speedup**[cite: 3].
3. **Hardware Constraints & Bottlenecks:** Although 12 threads were active, speedup scaling remained sub-linear. This performance gap is caused by shared CPU memory bus bandwidth limits and cache degradation due to column-wise access strides in matrix $B$ ($B[k \cdot N + j]$)[cite: 2].
