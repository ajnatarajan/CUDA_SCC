Ajay Natarajan and Eugene Shao's Final Project CS 179 Spring 2020
annatara@caltech.edu
eyshao@caltech.edu

USAGE INSTRUCTIONS:

make clean
make
./kosaraju input1_gpu.txt input1_cpu.txt input2_gpu.txt input2_cpu.txt 3,17000,0.8_GPU.txt 3,17000,0.8_CPU.txt

Running the above set of commands is the demo.
The output when run on Titan for us is:

annatara@titan:~/FinalProject$ ./kosaraju input1_gpu.txt input1_cpu.txt
input2_gpu.txt input2_cpu.txt 3,17000,0.8_GPU.txt 3,17000,0.8_CPU.txt
GPU answer: 3 computed in 5.7808 milliseconds.
CPU answer: 3 computed in 0.064544 milliseconds.
GPU answer: 1 computed in 2.03267 milliseconds.
CPU answer: 1 computed in 0.031584 milliseconds.
GPU answer: 6644 computed in 100792 milliseconds.
CPU answer: 6644 computed in 59.7827 milliseconds.

OUTPUT EXPLANATION:
The explanation for this is that the number represents the minimum number of
required connections for each testcase. Note that the CPU and GPU both output
the same number— if they did not, this would imply one or both of them were
incorrect. Every 2 lines corresponds to 1 testcase. In the first two test cases,
we deal with a very small number of airports and connections and thus the
required number of connections is fairly small. In fact, the second test case
involves a cycle between all other airports and then one isolated airport. Thus,
we need exactly 1 connection, and that is the output. The third test case, however,
involves 26**3 = 17576 airports with close to 17000 connections. This of course
will probably require significantly more connections, as evidenced by 6644
being the proper output.

Each testcase is represented by 2 files— one for CPU and one for GPU.
This is because the CPU code requires one format while the GPU code
requires another. We provide a Python file to generate new testcases if you guys
would like to generate your own random test cases.
You can pick:
(A) Length of each airport name— call this x. The number of airports will then
be 26**x.
(B) Number of pre-existing connections between airports (any positive integer).
Note, the number of actual connections may be a bit less than what you actually
enter. This is because we randomly generate these connections and if a duplicate
occurs, we don't add another connection.
(C) Probability for each connection to be intra-continental. (any value between
0 and 1).
Generate your own test cases by running:

python3 create_examples.py
Answer the prompted questions and txt files will appear in your directory
representing the CPU and GPU rules according to your entered values.

MOTIVATION:

This project adheres to the “parallelize an algorithm” choice, or more
specifically— an algorithm to solve a certain question will be developed, and
then this algorithm will be parallelized. The question at hand is a well-known
problem in the airline industry— the Interconnected Country Problem (ICP)— which
we now explain:

The ICP can be formalized as the following:
Given:
1. S, a starting airport
2. PORTS, a list of all airports
3. FLIGHTS, a list of pairs where each pair is of the form (airport #1, airport
#2) and represents a one-way flight from airport #1 to airport #2

Return:
1. NUM_ADD, the minimum number of one-way flights we must add to FLIGHTS to
ensure any traveller can get from S to any airport in the list PORTS via some
path. That is, for any airport P in PORTS, there exists a path S -> A -> B -> …
-> P (for some airports A, B, etc. in PORTS).

Determining the minimum number of one-way flights to add in order to ensure
seamless travel from some starting airport to all other airports is a critical
optimization problem. An airport will generally want to ensure a semblance of
connectedness to ensure it can service travelers from critical airports. For
example, one could imagine that LAX or JFK might be considered "starting" airports
in this problem as they are popular, widely-visited, and likely have the greatest
number of travelers leaving them for vacations. If an airline can find the
minimum number of flights they need to add to ensure any traveler leaving from
LAX can get to anywhere else, it will improve the chances that LAX travelers use
this airline, and sales increase resulting in business growth.

This problem ended up being much harder than anticipated as we ran into several
roadblocks, one of which is described in the next section.


HIGH-LEVEL OVERVIEW OF ALGORITHM

Our CPU algorithm is as follows:
1. Process PORTS and FLIGHTS to form a graph G, such that each airport A1 in
PORTS is represent by a vertex V1, and each pair (A1, A2) is represented by a
directed edge going from V1 to V2 (which are the representative vertices for A1,
 A2, respectively).
2. Identify Strongly Connected Components (SCCs) in this graph via Kosaraju’s
Algorithm. This involves 2 rounds of DFS.
3. Form a new graph G’ by compacting each SCC in G down to one vertex in G’.
Any directed edge going from some node in SCC #1 in G to some node in SCC #2 in
G is represented as a directed edge from SCC #1 to SCC #2 in G’.
4. Return the number of nodes in G’ with in-degree 0.

We parallelize steps 1, 3, and 4. However, after a literary search,
we found that while methods do exist to perform DFS on a DAG (directed acyclic
graph), no methods exist (or at least none that we could find) to perform DFS
on a graph with cycles. As a result, we had to alter our base algorithm
approach— one to involve BFS as opposed to DFS. The implemented GPU algorithm
then parallelizes a slightly modified CPU approach that relies on BFS as
opposed to DFS.

The GPU algorithm implements kernels for the following three sub-algorithms:
1. Trim: Removes vertices with either indegree 0 or outdegree 0 that form their
own SCC.
2. BFS: Does a BFS search from a starting airport using a given adjacency
matrix.
3. Find Degree Zero SCCs: Iterates through all SCCs in the graph and counts
the number of SCCs with indegree 0.

The final GPU algorithm is the following:
1. Process PORTS and FLIGHTS to form a graph G, such that each airport A1 in
A1 in PORTS is represent by a vertex V1, and each pair (A1, A2) is represented
by a directed edge going from V1 to V2 (which are the representative vertices
for A1, A2, respectively).
2. Trim the graph G.
3. Repeat the following steps until there are no points that have not been
assigned an SCC:
3a. Obtain a pivot vertex (one that has not yet been assigned to an SCC).
3b. Run a forward BFS on G from the pivot vertex using G's adjacency matrix.
3c. Run a backward BFS on G from the pivot vertex using G's reverse adjacency
matrix.
3d. Obtain the intersection points of the vertices visited by the forward BFS in
step 3a and those visited by the backward BFS in step 3b. Assign these points
to an SCC.
3e. Trim the remaining vertices that have yet to be assigned to an SCC.
4. Find the number of SCCs that have degree zero that do not contain the
starting airport. This number is the answer.


GPU OPTIMIZATION AND SPECIFICS:
The three algorithms were implemented as GPU kernels in the following manner:

Trim: It does so by looping through all vertices through the adjacency and
the reverse adjacency list and checking whether the vertex has indegree 0
(represented by all zeroes in its row in the reverse adjacency matrix) or
outdegree 0 (represented by all zeroes in its row in the adjacency matrix).
This looping is optimized through a GPU kernel as the loop does not require
sequential computation.
BFS kernel: Each vertex is assigned a GPU thread. The BFS algorithm starts with
a BFS kernel call in which the only vertex in the queue is the pivot vertex.
During each BFS kernel call, the frontier array is updated to contain the
unvisited neighbors of the vertices that are currently in the queue. Then, the
BFS helper function updates the queue with the frontier vertices and recalls the
BFS kernel with the updated values. This process continues until the frontier
array is empty, at which point the helper function returns the set of vertices
visited by the BFS kernel to the main function.
Find Degree Zero SCCs: Each vertex is assigned a GPU thread. This algorithm then
iterates through all vertices and determines whether the vertex has an incoming
edge from a vertex in another SCC. In the case that there does exist an incoming
edge and the destination vertex is not in the SCC containing the starting node,
we mark that the SCC of the destination vertex does not have indegree 0. In
order to avoid the edge case in which the start_port has indegree 0 and the
algorithm unwantingly adds one to our count, we start by setting a flag to
notify that the indegree of the SCC containing the start port is not zero.
Finally, the algorithm loops through all SCCs and adds 1 for each SCC that has
an indegree of 0.

The GPU code also contains one more GPU kernel that optimizes the computation
of the airport adjacency and reverse adjacency matrices. This optimization
relies on assigning each route a threadIdx and updating the adjacency and the
reverse adjacency matrices (represented as 1D arrays). Because no route relies
on one another, we can distribute the work of updating the adjacency matrices
to multiple threads, dramatically increasing the overall runtime.

CODE STRUCTURE:
Everything is in the main directory. From a GPU standpoint, we have 3 critical
files— kosaraju.cu, kosaraju.cuh, and primary.cpp. Primary.cpp has the main
function that is run to compute CPU and GPU times on a certain instance.
Kosaraju.cuh is the header file for kosaraju.cu which has all the GPU
optimizations. create_examples.py allows you, the graders, to create your own
sample test cases as txt files.


PERFORMANCE ANALYSIS:
The GPU runs slower than the CPU on many of these cases— there are two reasons
for this.
1) Because our approach uses an adjacency matrix, Titan runs out of memory if I
try to test with larger and larger number of airports. Thus, I am only able to
test on relatively small sample sizes.
2) Our CPU approach relies on Kosaraju's algorithm which then relies on DFS.
Because no approach, according to our lit search, exists to parallelize DFS
on a directed graph with cycles, we had to change our base approach. To keep
the consistency with the CPU Demo submitted earlier, we did not change our CPU
approach, but we did have to change the approach upon which the GPU would
parallelize. Thus, call our CPU Demo approach "Algorithm 1". Call the new approach
that our GPU code is parallelizing and optimizing "Algorithm 2". When run on a
CPU, Algorithm 2 is much slower than Algorithm 1. Thus, the GPU code is parallelizing
a worse runtime algorithm than what the CPU code is doing itself. Thus, even with
the optimizations, the GPU will still run slower than the fastest CPU approach.
Thus, it would be an interesting case for further work to solve an approach to
parallelize DFS on directed graphs with cycles, as if this is solved, this would
be the gateway to creating a GPU approach in this scenario which is much faster
than the CPU approach.

SMALLER COMPONENTS OF THE PROJECT:
We note that the rubric says to provide instructions on how to run smaller
components of the project. However, for our implementation at least, there are
no smaller components of the project. The only way to run the process is to pass
in exactly 3 testcases (a total of 6 files, 2 files per instance for the CPU
and GPU rules each) as command line arguments. You can do this with our provided
files, or you can generate your own using create_examples.py as detailed in a
prior section.
