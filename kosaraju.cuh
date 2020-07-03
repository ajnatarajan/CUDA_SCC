#include <cufft.h>
#include <vector>
#include <map>
#include <unordered_map>
#include <stack>
#include <set>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

/* Copied over from CS 179 set 1 code for GPU Error Checking. */
#define gpuErrchk(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "gpu_assert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

/* These are just wrapper functions to call kernels. */


void cudaCallAirportAdjacencyKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        int *dev_routes,
        int *dev_adj,
        int *dev_radj,
        int n_ports,
        int n_routes
        );

void cudaCallTrimGraph(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        int *adj,
        int *row_sum,
        bool *mark,
        int n_ports
        );

void cudaCallBFSKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        int *adj,
        bool *visited,
        bool *dev_frontier,
        int start_port,
        int n_ports,
        int *dev_flag
        );

void cudaCallFindDegreeZeroSCCKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        int *dev_adj,
        int *dev_radj,
        int *dev_reps,
        int *dev_zeroes,
        int *dev_total,
        int start_port,
        int n_ports
        );
