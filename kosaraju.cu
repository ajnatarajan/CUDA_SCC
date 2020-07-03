#include <cstdio>

#include <cuda_runtime.h>
#include "kosaraju.cuh"

/* Fill out the adjacency list and the reverse adjacency list as according to
 * the routes given. Each route represents a directed edge.
 */
__global__
void cudaAirportAdjacencyKernel(int *dev_routes,
        int *dev_adj,
        int *dev_radj,
        int n_ports,
        int n_routes) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < n_routes) {
        int first = dev_routes[2 * i];
        int second = dev_routes[2 * i + 1];
        dev_adj[first * n_ports + second] = 1;
        dev_radj[second * n_ports + first] = 1;
        i += blockDim.x * gridDim.x;
    }
}
/* Wrapper function to call cudaAirportAdjacencyKernel. */
void cudaCallAirportAdjacencyKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        int *dev_routes,
        int *dev_adj,
        int *dev_radj,
        int n_ports,
        int n_routes) {
    cudaAirportAdjacencyKernel<<<blocks, threadsPerBlock>>>
        (dev_routes, dev_adj, dev_radj, n_ports, n_routes);
}

/* Remove any vertices with in-degree and out-degree 0, just for optimization. */
__global__
void cudaTrimGraph(int *m,
        int *row_sum,
        bool *mark,
        int n_ports) {
    // For i = 0 to n_ports - 1 inclusive, achieve the following:
    // row_sum[i] = sum from j = 0 to n_ports - 1 of m[i * n_ports + j] * !mark[j]
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < n_ports) {
        int total = 0;
        for (int j = 0; j < n_ports; j++) {
            total += m[i * n_ports + j] * !(mark[j]);
        }
        row_sum[i] = total;
        i += blockDim.x * gridDim.x;
    }
}

void cudaCallTrimGraph(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        int *adj,
        int *row_sum,
        bool *mark,
        int n_ports) {
    cudaTrimGraph<<<blocks, threadsPerBlock>>>(adj, row_sum, mark, n_ports);
}

__global__
void cudaBFSKernel(int *adj,
        bool *frontier,
        bool *visited,
        int n_ports) {
    // Do the BFS search
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < n_ports) {
        if (frontier[tid]) {
            frontier[tid] = false;
            visited[tid] = true;
            for (int i = 0; i < n_ports; i++) {
                if (adj[tid * n_ports + i] && !visited[i]) {
                    frontier[i] = true;
                }
            }
        }
        tid += blockDim.x * gridDim.x;
    }
}

/* Returns whether the frontier array contains any true values. */
__global__
void cudaContainsTrueKernel(bool *frontier,
        int *dev_flag,
        int n_ports) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < n_ports) {
        if (frontier[tid]) {
            dev_flag[0] *= 0;
        }
        tid += blockDim.x * gridDim.x;
    }
}

/* Wrapper function to perform BFS. */
void cudaCallBFSKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        int *adj,
        bool *visited,
        bool *dev_frontier,
        int start_port,
        int n_ports,
        int *dev_flag) {
    int *flag = (int *) malloc(sizeof(int));
    while (true) {
        for (int i = 0; i < n_ports; i++) {
            cudaBFSKernel<<<blocks, threadsPerBlock>>>
                (adj, dev_frontier, visited, n_ports);
        }
        cudaContainsTrueKernel<<<blocks, threadsPerBlock>>>
            (dev_frontier, dev_flag, n_ports);
        cudaMemcpy(flag, dev_flag, sizeof(int), cudaMemcpyDeviceToHost);
        if (flag[0]) {
            break;
        }
    }
    free(flag);
}

/* Fill out an array, one value for each airport. If an index i is some
 * representative node of an SCC (that is not the starting airport) and we have
 * that dev_zeroes[i] = 0 at the end of this kernel, then that means that
 * index represents an airport who is a representative node of an SCC that has
 * no incoming edges.
 */
__global__
void cudaFindDegreeZeroSCCKernel(int *adj,
        int *radj,
        int *reps,
        int *dev_zeroes,
        int start_port,
        int n_ports) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    dev_zeroes[start_port] = 1;
    while (i < n_ports) {
        unsigned int curr_rep = reps[i];
        for(int j = 0; j < n_ports; j++) {
            if (radj[i * n_ports + j] == 1 && reps[j] != curr_rep) {
                dev_zeroes[curr_rep] = 1;
                break;
            }
        }
        i += blockDim.x * gridDim.x;
    }
}

/* Find number of representative nodes that have in-degree 0 (excluding
 * starting airport). This is then the final answer to our algorithm.
 */
__global__
void cudaFindAllZeroesKernel(int *dev_reps,
        int *dev_zeroes,
        int *dev_total,
        int n_ports){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < n_ports) {
        if(dev_reps[i] == i && dev_zeroes[i] == 0) {
            atomicAdd(dev_total, 1);
        }
        i += blockDim.x * gridDim.x;
    }
}

/* Wrapper function to call cudaFindDegreeZeroSCCKernel and
 * cudaFindAllZeroesKernel.
 */
void cudaCallFindDegreeZeroSCCKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        int *dev_adj,
        int *dev_radj,
        int *dev_reps,
        int *dev_zeroes,
        int *dev_total,
        int start_port,
        int n_ports) {
    cudaFindDegreeZeroSCCKernel<<<blocks, threadsPerBlock>>>
        (dev_adj, dev_radj, dev_reps, dev_zeroes, start_port, n_ports);

    cudaFindAllZeroesKernel<<<blocks, threadsPerBlock>>>
        (dev_reps, dev_zeroes, dev_total, n_ports);
}
