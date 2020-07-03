NVCC=nvcc
CUDAFLAGS= -arch=sm_30
OPT= -g -G
RM=/bin/rm -f

all: kosaraju

main: kosaraju.o primary.o
				${NVCC} ${OPT} -o main kosaraju.o primary.o

primary.o: kosaraju.cuh primary.cpp
				${NVCC} ${OPT} ${CUDAFLAGS} -std=c++11 -c primary.cpp

kosaraju.o: kosaraju.cuh kosaraju.cu
				$(NVCC) ${OPT} $(CUDAFLAGS)        -std=c++11 -c kosaraju.cu

kosaraju: kosaraju.o primary.o
				${NVCC} ${CUDAFLAGS} -o kosaraju kosaraju.o primary.o

clean:
				${RM} *.o kosaraju
