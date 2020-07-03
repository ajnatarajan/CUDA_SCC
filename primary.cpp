#include <vector>
#include <map>
#include <unordered_map>
#include <stack>
#include <set>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <cassert>
#include "kosaraju.cuh"



using namespace std;

/*
 * ==================================
 *          BEGIN CPU CODE
 * ==================================
 */

/*
 * Function to create a dictionary that maps each airport name from an input
 * list of airport names to a unique identifier integer.
 */
unordered_map<string, int> convert_ports_to_ints(vector<string> ports) {
    unordered_map<string, int> pokedex;
    int counter = 0;
    for (string port : ports) {
        pokedex[port] = counter;
        ++counter;
    }
    return pokedex;
}

/*
 * DFS helper function that helps the DFS through the graph.
 */
void dfs_flights_helper(stack<int>& port_stack, vector<vector<int>>& connect, vector<bool>& visited, int curr) {
    for (int neighbor : connect[curr]) {
        if (!visited[neighbor]) {
            visited[neighbor] = true;
            dfs_flights_helper(port_stack, connect, visited, neighbor);
        }
    }
    port_stack.push(curr);
}

/*
 * DFS function that serves as the mechanism to DFS through the graph.
 */
void dfs_flights(stack<int>& port_stack, vector<vector<int>>& connect, vector<bool>& visited) {
  for (int i = 0; i < visited.size(); ++i) {
    if (!visited[i]) {
      visited[i] = true;
      dfs_flights_helper(port_stack, connect, visited, i);
    }
  }
}

/*
 * Create the reverse adjacency matrix. Note that the reverse adjacency matrix
 * is the transpose of the normal adjacency matrix.
 */
vector<vector<int>> transpose(vector<vector<int>> connect) {
  vector<vector<int>> res(connect.size());
  for (int i = 0; i < connect.size(); ++i) {
    vector<int> adj = connect[i];
    for (int neighbor : adj) {
      res[neighbor].push_back(i);
    }
  }
  return res;
}

/*
 * Helper function for the second DFS traversal through the graph.
 */
void dfs_flights2_helper(vector<int>& rep, vector<bool>& visited2, vector<vector<int>>& connect_T, int curr, int count) {
  rep[curr] = count;
  for (int neighbor : connect_T[curr]) {
    if (!visited2[neighbor]) {
      visited2[neighbor] = true;
      dfs_flights2_helper(rep, visited2, connect_T, neighbor, count);
    }
  }
}

/*
 * The second DFS traversal through the graph. This traversal aims to assign
 * each vertex to a unique SCC.
 */
int dfs_flights2(vector<int>& rep, vector<bool>& visited2, vector<vector<int>>& connect_T, int curr, int count) {
  visited2[curr] = true;
  dfs_flights2_helper(rep, visited2, connect_T, curr, count);
}

/*
 * Wrapper function to run the kosaraju algorithm to solve the ICP.
 */
int run_kosaraju(vector<string> ports, vector<vector<string>> flights, string starting_airport) {
  int num_add = 0;
  // Call DFS(G)
  // Call DFS(G_T) in decreasing order of their finish times
  // Vertices as separate SCCs

  // pokedex so we use ints
  unordered_map<string, int> pokedex = convert_ports_to_ints(ports);
  int n_cities = pokedex.size();
  int start_index = pokedex[starting_airport];

  // Create the graph, we choose adjacency lists
  vector<vector<int>> connect(n_cities);
  for (vector<string> flight_pair : flights) {
    int first, second;
    first = pokedex[flight_pair[0]];
    second = pokedex[flight_pair[1]];
    connect[first].push_back(second);
  }

  // 1) Create a stack
  stack<int> port_stack;

  // 2) dfs on the graph
  vector<bool> visited(n_cities, false);
  dfs_flights(port_stack, connect, visited);

  // 3) Transpose the graph
  vector<vector<int>> connect_T = transpose(connect);

  // 4) Do the popping thing
  vector<bool> visited2(n_cities, false);
  vector<int> rep(n_cities);
  int n_scc = 0;
  while (port_stack.size() > 0) {
    int city = port_stack.top();
    port_stack.pop();
    if (!visited2[city]) {
      dfs_flights2(rep, visited2, connect_T, city, n_scc);
      ++n_scc;
    }
  }

  vector<bool> deg(n_scc, true);

  // 5) Figure out degree of each scc
  for (int i = 0; i < n_cities; ++i) {
    for (int neighbor : connect[i]) {
      if (rep[i] != rep[neighbor]) {
        deg[rep[neighbor]] = false;
      }
    }
  }

  // 6) Identify degree 0 scc
  for (int i = 0; i < deg.size(); ++i) {
    if (deg[i] && rep[start_index] != i) {
      ++num_add;
    }
  }

  return num_add;
}

/*
 * ==================================
 *           END CPU CODE
 * ==================================
 */




/*
 * Removes vertices with either indegree 0 or outdegree 0, as that implies they
 * are in their own SCC.
 */
void trim(int blocks, int threadsPerBlock,
    int *dev_adj, int *dev_radj,
    int *dev_row_sum, bool *dev_mark,
    int *row_sum, bool *mark, int *reps,
    int n_ports) {
    gpuErrchk(cudaMemcpy(dev_mark, mark, n_ports * sizeof(bool), cudaMemcpyHostToDevice));

    // Outdegree 0
    gpuErrchk(cudaMemset(dev_row_sum, 0, n_ports * sizeof(int)));
    cudaCallTrimGraph(blocks, threadsPerBlock,
        dev_adj, dev_row_sum, dev_mark, n_ports);

    gpuErrchk(cudaMemcpy(row_sum, dev_row_sum, n_ports * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n_ports; i++) {
        if (!mark[i] && row_sum[i] == 0) {
            reps[i] = i;
            mark[i] = true;
        }
    }


    // Indegree 0
    gpuErrchk(cudaMemset(dev_row_sum, 0, n_ports * sizeof(int)));
    cudaCallTrimGraph(blocks, threadsPerBlock,
        dev_radj, dev_row_sum, dev_mark, n_ports);

    gpuErrchk(cudaMemcpy(row_sum, dev_row_sum, n_ports * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n_ports; i++) {
        if (!mark[i] && row_sum[i] == 0) {
            reps[i] = i;
            mark[i] = true;
        }
    }

}

/*
 * Gets a vertex that has not been assigned to an SCC yet. Returns -1 if all
 * vertices have already been assigned an SCC.
 */
int getPivot(bool *mark, int n_ports) {
    int pivot = 0;
    for (int i = 0; i < n_ports; i++) {
        if (mark[pivot]) {
            pivot++;
        } else {
            return pivot;
        }
    }
    return -1;
}

/*
 * Solution algorithm.
 */
int macroAlgorithm(int *airports, int *routes,
    int start_port, int n_ports, int n_routes) {
    // Setting initial values.
    int blocks = 512;
    int threadsPerBlock = 1024;

    int *adj;
    int *radj;

    int *dev_ports;
    int *dev_routes;
    int *dev_adj;
    int *dev_radj;

    // Mallocing memory on the CPU for data structures.
    adj = (int *) malloc(n_ports * n_ports * sizeof(int));
    radj = (int *) malloc(n_ports * n_ports * sizeof(int));

    // Mallocing memory on the GPU for data structures on the device.
    gpuErrchk(cudaMalloc((void **)&dev_ports, n_ports * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&dev_routes, n_routes * 2 * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&dev_adj, n_ports * n_ports * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&dev_radj, n_ports * n_ports * sizeof(int)));

    // Setting the correct values for data structures on the GPU.
    gpuErrchk(cudaMemcpy(dev_ports, airports, n_ports * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_routes, routes, n_routes * 2 * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemset(dev_adj, 0, n_ports * n_ports * sizeof(int)));
    gpuErrchk(cudaMemset(dev_radj, 0, n_ports * n_ports * sizeof(int)));

    // GPU call to determine the adjacency matrix of the graph.
    cudaCallAirportAdjacencyKernel(blocks, threadsPerBlock,
        dev_routes, dev_adj, dev_radj, n_ports, n_routes);

    // Transferring results from the airport adjacency kernel to the host.
    gpuErrchk(cudaMemcpy(adj, dev_adj, n_ports * n_ports * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(radj, dev_radj, n_ports * n_ports * sizeof(int), cudaMemcpyDeviceToHost));

    // SCC Decomposition
    int pivot;
    int *reps;
    int *row_sum;
    bool *mark;
    bool *visited_fwd;
    bool *visited_bwd;
    bool *frontier;
    int *flag;
    int *total;
    int *zeroes;

    int *dev_reps;
    int *dev_row_sum;
    bool *dev_mark;
    bool *dev_visited_fwd;
    bool *dev_visited_bwd;
    int *dev_zeroes;
    bool *dev_frontier;
    int *dev_flag;
    int *dev_total;

    // Mallocing memory for host data structures.
    row_sum = (int *) malloc(n_ports * sizeof(int));
    reps = (int *) malloc(n_ports * sizeof(int));
    mark = (bool *) malloc(n_ports * sizeof(bool));
    visited_fwd = (bool *) malloc(n_ports * sizeof(bool));
    visited_bwd = (bool *) malloc(n_ports * sizeof(bool));
    frontier = (bool *) malloc(n_ports * sizeof(bool));
    flag = (int *) malloc(sizeof(int));
    total = (int *) malloc(sizeof(int));
    zeroes = (int *) malloc(n_ports * sizeof(int));

    // Set memory for host variables to default values.
    memset(row_sum, 0, n_ports * sizeof(int));
    memset(reps, 0, n_ports * sizeof(int));
    memset(mark, false, n_ports * sizeof(bool));
    memset(visited_fwd, false, n_ports * sizeof(bool));
    memset(visited_bwd, false, n_ports * sizeof(bool));

    // Malloc space on the device for all necessary variables.
    gpuErrchk(cudaMalloc((void **)&dev_reps, n_ports * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&dev_row_sum, n_ports * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&dev_mark, n_ports * sizeof(bool)));
    gpuErrchk(cudaMalloc((void **)&dev_visited_fwd, n_ports * sizeof(bool)));
    gpuErrchk(cudaMalloc((void **)&dev_visited_bwd, n_ports * sizeof(bool)));
    gpuErrchk(cudaMalloc((void **)&dev_zeroes, n_ports * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&dev_frontier, n_ports * sizeof(bool)));
    gpuErrchk(cudaMalloc((void **)&dev_flag, sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&dev_total, sizeof(int)));

    // Perform a trim for optimization.
    trim(blocks, threadsPerBlock, dev_adj, dev_radj,
        dev_row_sum, dev_mark, row_sum, mark, reps, n_ports);

    pivot = getPivot(mark, n_ports);
    flag[0] = 1;
    total[0] = 0;
    while (pivot > -1) {
        // Forward search
        gpuErrchk(cudaMemset(dev_visited_fwd, false, n_ports * sizeof(bool)));
        memset(frontier, false, n_ports * sizeof(bool));
        frontier[pivot] = true;
        gpuErrchk(cudaMemcpy(dev_flag, flag, sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dev_frontier, frontier, n_ports * sizeof(bool), cudaMemcpyHostToDevice));

        cudaCallBFSKernel(blocks, threadsPerBlock, dev_adj, dev_visited_fwd,
            dev_frontier, pivot, n_ports, dev_flag);

        // Backward search
        gpuErrchk(cudaMemset(dev_visited_bwd, false, n_ports * sizeof(bool)));
        memset(frontier, false, n_ports * sizeof(bool));
        frontier[pivot] = true;
        gpuErrchk(cudaMemcpy(dev_flag, flag, sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dev_frontier, frontier, n_ports * sizeof(bool), cudaMemcpyHostToDevice));

        cudaCallBFSKernel(blocks, threadsPerBlock, dev_radj, dev_visited_bwd,
            dev_frontier, pivot, n_ports, dev_flag);

        // Copy from device to host for visited_fwd and visited_bwd
        gpuErrchk(cudaMemcpy(visited_fwd, dev_visited_fwd, n_ports * sizeof(bool), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(visited_bwd, dev_visited_bwd, n_ports * sizeof(bool), cudaMemcpyDeviceToHost));

        // update
        for (int i = 0; i < n_ports; i++) {
            if (visited_fwd[i] && visited_bwd[i]) {
                reps[i] = pivot;
                mark[i] = true;
            }
        }

        // trim
        trim(blocks, threadsPerBlock, dev_adj, dev_radj,
            dev_row_sum, dev_mark, row_sum, mark, reps, n_ports);

        // get pivot
        pivot = getPivot(mark, n_ports);
    }

    // Determine indegree 0 -- note set starting city as 0
    gpuErrchk(cudaMemcpy(dev_total, total, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_reps, reps, n_ports * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(dev_zeroes, 0, n_ports * sizeof(int)));

    cudaCallFindDegreeZeroSCCKernel(blocks, threadsPerBlock,
        dev_adj, dev_radj, dev_reps, dev_zeroes, dev_total, start_port, n_ports);

    // Copy dev_total over to total so we can access the value computed by the
    // GPU code on our host machine.
    gpuErrchk(cudaMemcpy(total, dev_total, sizeof(int), cudaMemcpyDeviceToHost));

    int res = total[0];

    // For some reason, cudaFree here results in errors, while if we do not
    // cudaFree, everything works perfectly. Thus, we remove it.
    /*
    gpuErrchk(cudaFree(dev_ports));
    gpuErrchk(cudaFree(dev_routes));
    gpuErrchk(cudaFree(dev_adj));
    gpuErrchk(cudaFree(dev_radj));
    gpuErrchk(cudaFree(dev_row_sum));
    gpuErrchk(cudaFree(dev_mark));
    gpuErrchk(cudaFree(dev_visited_fwd));
    gpuErrchk(cudaFree(dev_visited_bwd));
    gpuErrchk(cudaFree(dev_zeroes));
    gpuErrchk(cudaFree(dev_frontier));
    gpuErrchk(cudaFree(dev_flag));
    */

    // Free all resources used.
    free(adj);
    free(radj);
    free(reps);
    free(row_sum);
    free(mark);
    free(visited_fwd);
    free(visited_bwd);
    free(flag);
    free(total);
    free(zeroes);

    return res;
}

int main(int argc, char *argv[]) {
    if (argc != 7) {
        printf("Incorrect usage, enter ./kosaraju <GPU filename> <CPU filename>\n");
        return 1;
    }

    for(int argnum = 0; argnum < 3; argnum++) {
        int start_port, n_ports, n_routes;
        string filename, port;
        map<string, int> airport_ids;
        int *airports;
        int *routes;

        // File will include two integers n_ports, n_routes in the first line which
        // equal the number of airports and the number of routes that follow in the
        // lines below.

        ifstream ifs(argv[1 + 2*argnum], ifstream::in);
        if(!ifs) {
            printf("Could not find or open file\n");
            return 1;
        }

        ifs >> n_ports >> n_routes;
        airports = (int *) malloc(n_ports * sizeof(int));
        routes = (int *) malloc(n_routes * 2 * sizeof(int));

        for (int i = 0; i < n_ports; i++) {
            ifs >> port;
            airports[i] = i;
            airport_ids[port] = i;
        }

        for (int i = 0; i < n_routes; i++) {
            for (int j = 0; j < 2; j++) {
                ifs >> port;
                routes[2 * i + j] = airport_ids[port];
            }
        }
        ifs >> port;
        start_port = airport_ids[port];

        cudaEvent_t start;
        cudaEvent_t stop;

        #define START_TIMER() {                         \
              gpuErrchk(cudaEventCreate(&start));       \
              gpuErrchk(cudaEventCreate(&stop));        \
              gpuErrchk(cudaEventRecord(start));        \
        }

        #define STOP_RECORD_TIMER(name) {                           \
              gpuErrchk(cudaEventRecord(stop));                     \
              gpuErrchk(cudaEventSynchronize(stop));                \
              gpuErrchk(cudaEventElapsedTime(&name, start, stop));  \
              gpuErrchk(cudaEventDestroy(start));                   \
              gpuErrchk(cudaEventDestroy(stop));                    \
        }

        float gpu_time = -1;
        float cpu_time = -1;
        START_TIMER();
        int res = macroAlgorithm(airports, routes, start_port, n_ports, n_routes);
        STOP_RECORD_TIMER(gpu_time);

        cout << "GPU answer: " << res << " computed in " << gpu_time << " milliseconds.\n";
        free(airports);
        free(routes);
        ifs.close();





        /*
         * CPU TESTING
         */

        ifstream ifs2(argv[2+2*argnum], ifstream::in);
        if(!ifs2) {
            printf("Could not find or open file\n");
            return 1;
        }

        vector<string> cpu_airports;
        vector<vector<string>> cpu_routes;
        string starting_airport;

        int num_airports;
        ifs2 >> num_airports;
        for(int j = 0; j < num_airports; j++) {
            string curr;
            ifs2 >> curr;
            cpu_airports.push_back(curr);
        }

        ifs2 >> starting_airport;

        int num_connections;
        ifs2 >> num_connections;
        for(int j = 0; j < num_connections; j++) {
            string start;
            string end;
            ifs2 >> start;
            ifs2 >> end;
            cpu_routes.push_back({start, end});
        }

        START_TIMER();
        int ans = run_kosaraju(cpu_airports, cpu_routes, starting_airport);
        STOP_RECORD_TIMER(cpu_time);
        cout << "CPU answer: " << ans << " computed in " << cpu_time << " milliseconds.\n";
        ifs2.close();
    }

    return 1;
}
