# Experiment 1 — Parallel Matrix Multiplication

A comparative implementation of **4000 × 4000 matrix multiplication** using four parallel/computational approaches:

- **Sequential C**
- **OpenMP**
- **MPI**
- **CUDA**

The project records execution time and verifies the result using `C[0][0] = 4000.00`.

> **Academic project:** Parallel Computing — Experiment 1

## 1. Objective

The objective of this experiment is to implement matrix multiplication using different execution models and compare their observed performance.

For matrices `A` and `B` of size `4000 × 4000`, initialized with `1.0`:

\[
C = A \times B
\]

Every element of the resulting matrix is therefore expected to be:

\[
4000.00
\]

The implementation verifies this using:

```text
Verification C[0][0] = 4000.00
```

## 2. Implementations

| Implementation | Technology | Configuration | Recorded Time |
|---|---|---:|---:|
| Sequential | C / GCC | 1 CPU execution | 204.398481 s |
| OpenMP | C / OpenMP | 8 threads | 43.578664 s |
| MPI | C / MPI | 4 processes | 60.802443 s |
| CUDA | CUDA C++ | 16 × 16 threads/block | 0.080268 s total phase* |

\* The CUDA result records the total CUDA phase from the successful run. Kernel execution was approximately `0.067203 s`.

These values are **observed experimental results**, not hardware-independent benchmarks. Timing can vary with CPU, RAM, GPU, compiler, MPI configuration, operating-system load, and measurement methodology.

## 3. Repository Structure

```text
Experiment-1-Matrix-Multiplication/
│
├── README.md
├── .gitignore
│
├── code/
│   ├── sequential/
│   │   └── matrix_sequential.c
│   │
│   ├── openmp/
│   │   └── matrix_openmp.c
│   │
│   ├── mpi/
│   │   ├── matrix_mpi.c
│   │   └── hosts.example
│   │
│   └── cuda/
│       └── matrix_cuda.cu
│
├── evidence/
│   ├── sequential/
│   │   └── 01_sequential_execution.png
│   │
│   ├── openmp/
│   │   └── 01_openmp_execution.png
│   │
│   ├── mpi/
│   │   ├── 01_mpi_execution.txt
│   │   └── 02_mpi_hostfile_note.txt
│   │
│   └── cuda/
│       └── 01_cuda_execution.txt
│
└── results/
    ├── performance_summary.md
    └── results.csv
```

## 4. Sequential Implementation

### Source

`code/sequential/matrix_sequential.c`

### Compile

```bash
cd code/sequential
gcc -O2 matrix_sequential.c -o matrix_sequential
```

### Run

```bash
./matrix_sequential
```

### Recorded result

```text
Sequential Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Execution Time = 204.398481 seconds
Verification C[0][0] = 4000.00
```

Evidence:

`evidence/sequential/01_sequential_execution.png`

## 5. OpenMP Implementation

### Source

`code/openmp/matrix_openmp.c`

### Compile

```bash
cd code/openmp
gcc -O2 -fopenmp matrix_openmp.c -o matrix_openmp
```

### Run

```bash
./matrix_openmp
```

The recorded experiment used **8 OpenMP threads**.

### Recorded result

```text
OpenMP Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Number of Threads Used = 8
Execution Time = 43.578664 seconds
Verification C[0][0] = 4000.00
```

Evidence:

`evidence/openmp/01_openmp_execution.png`

## 6. MPI Implementation

### Source

`code/mpi/matrix_mpi.c`

### Requirements

- MPI implementation such as Open MPI
- GCC / `mpicc`
- A configured master/worker environment for multi-node execution
- SSH access between participating nodes

### Compile

```bash
cd code/mpi
mpicc -O2 matrix_mpi.c -o matrix_mpi
```

### Hostfile

Use `hosts.example` as the template:

```text
master slots=1
worker1 slots=1
worker2 slots=1
worker3 slots=1
```

Create the actual `hosts` file for your environment. Do **not** commit private/environment-specific IP addresses unless they are intentionally part of the assignment.

### Run

```bash
mpirun -np 4 --hostfile hosts ./matrix_mpi
```

### Recorded result

```text
MPI Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Number of MPI Processes = 4
Execution Time = 60.802443 seconds
Verification C[0][0] = 4000.00
```

### Important MPI evidence note

The recorded terminal output from the experiment showed four MPI ranks, but the displayed hostnames included rank 3 on `worker1`. Therefore, before presenting the MPI result as a clean four-node experiment, verify the hostfile and ensure the intended allocation is:

```text
Rank 0 -> master
Rank 1 -> worker1
Rank 2 -> worker2
Rank 3 -> worker3
```

The repository includes the hostfile template specifically to keep the GitHub project professional and independent of one particular lab network.

## 7. CUDA Implementation

### Source

`code/cuda/matrix_cuda.cu`

### Requirements

- NVIDIA CUDA-capable GPU
- NVIDIA driver
- CUDA Toolkit
- `nvcc`
- A compatible Microsoft Visual C++ build environment on Windows

### Compile

From a CUDA-enabled Visual Studio Developer Command Prompt:

```powershell
cd C:\parallel_lab\cuda
nvcc -O2 matrix_cuda.cu -o matrix_cuda.exe
```

### Run

```powershell
.\matrix_cuda.exe
```

### Recorded result

```text
CUDA Matrix Multiplication Completed
Matrix Size = 4000 x 4000
Grid Size = 250 x 250 blocks
Block Size = 16 x 16 threads
Kernel Execution Time = 0.067203 seconds
Total CUDA Phase Time = 0.080268 seconds
Verification C[0][0] = 4000.00
```

The CUDA implementation uses a `16 × 16` thread block configuration and a `250 × 250` grid for a `4000 × 4000` matrix.

## 8. Performance Summary

### Recorded execution times

| Method | Time (s) | Verification |
|---|---:|---:|
| Sequential | 204.398481 | 4000.00 |
| OpenMP (8 threads) | 43.578664 | 4000.00 |
| MPI (4 processes) | 60.802443 | 4000.00 |
| CUDA (total phase) | 0.080268 | 4000.00 |

### Recorded speedup relative to sequential

Using:

\[
Speedup = \frac{T_{sequential}}{T_{method}}
\]

the recorded values are approximately:

- OpenMP: **4.69×**
- MPI: **3.36×**
- CUDA total phase: **2546.45×**

These are comparisons of the recorded runs. They should not be interpreted as universal performance guarantees because the four implementations use different execution models and timing scopes.

## 9. Result Verification

All four implementations use the same mathematical workload:

```text
Matrix Size = 4000 x 4000
A[i][j] = 1.0
B[i][j] = 1.0
Expected C[0][0] = 4000.00
```

The recorded successful executions verify:

```text
C[0][0] = 4000.00
```

This confirms that the computed result matches the expected value for the tested matrix initialization.

## 10. Build Environment

The experiments were performed across Linux/WSL/Ubuntu and Windows CUDA environments.

Typical tools used:

| Component | Tool |
|---|---|
| Sequential C | GCC |
| OpenMP | GCC + OpenMP |
| MPI | Open MPI + `mpicc` |
| CUDA | CUDA Toolkit + `nvcc` |
| MPI execution | `mpirun` |
| Repository | Git + GitHub |

## 11. Reproducibility

Clone the repository:

```bash
git clone https://github.com/PoojaKD07/Experiment-1-Matrix-Multiplication.git
cd Experiment-1-Matrix-Multiplication
```

Then follow the build/run commands in the corresponding implementation section.

For MPI, configure the local `hosts` file from `code/mpi/hosts.example` according to the actual lab network.

For CUDA, run the CUDA source on a machine with an NVIDIA GPU and a compatible CUDA/MSVC environment.

## 12. Evidence

Execution evidence is stored separately from source code:

- `evidence/sequential/` — sequential execution screenshot
- `evidence/openmp/` — OpenMP execution screenshot
- `evidence/mpi/` — MPI recorded output and configuration note
- `evidence/cuda/` — CUDA recorded output

The evidence directories are kept separate so the source tree remains clean and easy to review.

## 13. Conclusion

The experiment demonstrates the performance differences between sequential execution, shared-memory parallelism with OpenMP, distributed-memory parallelism with MPI, and GPU acceleration with CUDA.

For the recorded `4000 × 4000` workload:

- Sequential execution provided the baseline.
- OpenMP substantially reduced execution time using 8 threads.
- MPI distributed the workload across 4 processes.
- CUDA provided the lowest recorded execution time on the available GPU system.

The results demonstrate the importance of selecting an execution model appropriate to the available hardware and problem size.

---

**Author:** Pooja K D  
**Experiment:** Parallel Computing — Experiment 1: Matrix Multiplication
