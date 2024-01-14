gomp-only: main-openmp-only.o coordReader.o
	gcc main-openmp-only.o coordReader.o -o gomp-only -fopenmp -lm

gcomplete: main-mpi.o coordReader.o
	mpicc main-mpi.o coordReader.o -o gcomplete -fopenmp -lm

iomp-only: main-openmp-only.o coordReader.o
	icc main-openmp-only.o coordReader.o -o iomp-only -fopenmp -lm

icomplete: main-mpi.o coordReader.o
	mpiicc main-mpi.o coordReader.o -o icomplete -fopenmp -lm

main-openmp-only.o: main-openmp-only.c
	gcc -c main-openmp-only.c -fopenmp -lm

main-mpi.o: main-mpi.c
	mpicc -c main-mpi.c -fopenmp -lm

coordReader.o: coordReader.c
	gcc -c coordReader.c
