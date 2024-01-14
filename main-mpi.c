#include <mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<math.h>
#include<string.h>
#include<omp.h>

int readNumOfCoords(char *filename);
double **readCoords(char *filename, int numOfCoords);
void *writeTourToFile(int *tour, int tourLength, char *filename);
double **createDistanceMatrix(double **coords, int numOfCoords);
int *nearestAddition(double **dMatrix, int numOfCoords,int startVertex);
int* cheapestInsertion(double **dMatrix, int numOfCoords, int startVertex);
int* farthestInsertion(double **dMatrix, int numOfCoords, int startVertex);


int main(int argc, char *argv[]){
    
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if(argc != 5){
        printf("Program should be called as ./program <coordFile> <outFileName1> <outFileName2> <outFileName3>\n");
        MPI_Finalize();
        return 1;
    }

    char filename[500];
	char outFileName1[500];
    char outFileName2[500];
    char outFileName3[500];

	strcpy(filename, argv[1]);
	strcpy(outFileName1, argv[2]);
	strcpy(outFileName2, argv[3]);
	strcpy(outFileName3, argv[4]);


	//Reading files and setting up the distance matrix
	int numOfCoords = readNumOfCoords(filename);
	double **coords = readCoords(filename, numOfCoords);

	double tStart = omp_get_wtime();

    double **dMatrix = createDistanceMatrix(coords, numOfCoords);
    
    double minTourCost1 = __DBL_MAX__;
    double minTourCost2 = __DBL_MAX__;
    double minTourCost3 = __DBL_MAX__;
    int *bestTour1 = NULL;
    int *bestTour2 = NULL;
    int *bestTour3 = NULL;
    double globalMinTourCost1;
    double globalMinTourCost2;
    double globalMinTourCost3;
    // printf("numOfCoords: %d\n", numOfCoords);
    int start = numOfCoords * world_rank / world_size;
    int end = numOfCoords * (world_rank + 1) / world_size;

    for(int v = start; v < end; v++) {
		// printf("Process %d is working on vertex %d\n", world_rank, v);
        int* currentTour1, *currentTour2, *currentTour3;
        currentTour1 = farthestInsertion(dMatrix, numOfCoords, v);
        currentTour2 = cheapestInsertion(dMatrix, numOfCoords, v);
        currentTour3 = nearestAddition(dMatrix, numOfCoords, v);
        // currentTour[2] = nearestAddition(dMatrix, numOfCoords, v);
        // printf("Here1\n");
        // int* currentTour = farthestInsertion(dMatrix, numOfCoords, v);
        // Calculate the cost of the current tour
        double currentTourCost;
        currentTourCost = 0;
        for(int j = 0; j < numOfCoords; j++) {
            // printf("%d %d %d %d\n", i,j,currentTour[i][j],currentTour[i][j+1]);
            currentTourCost += dMatrix[currentTour1[j]][currentTour1[j+1]];
        }
        currentTourCost = currentTourCost + v*0.0000001;
        printf("Current cost : %f Vertex : %d\n", currentTourCost,v);
        if(currentTourCost < minTourCost1) {
            minTourCost1 = currentTourCost;
            // free(bestTour[i]); // Free the previous best tour
            bestTour1 = currentTour1;
        } else {
            // free(currentTour[i]);
        }
        currentTourCost = 0;
        for(int j = 0; j < numOfCoords; j++) {
            // printf("%d %d %d %d\n", i,j,currentTour[i][j],currentTour[i][j+1]);
            currentTourCost += dMatrix[currentTour2[j]][currentTour2[j+1]];
        }
        currentTourCost = currentTourCost + v*0.0000001;
        if(currentTourCost < minTourCost2) {
            minTourCost2 = currentTourCost;
            // free(bestTour[i]); // Free the previous best tour
            bestTour2 = currentTour2;
        } else {
            // free(currentTour[i]);
        }
        currentTourCost = 0;
        for(int j = 0; j < numOfCoords; j++) {
            // printf("%d %d %d %d\n", i,j,currentTour[i][j],currentTour[i][j+1]);
            currentTourCost += dMatrix[currentTour3[j]][currentTour3[j+1]];
        }
        currentTourCost = currentTourCost + v*0.0000001;
        if(currentTourCost < minTourCost3) {
            minTourCost3 = currentTourCost;
            // free(bestTour[i]); // Free the previous best tour
            bestTour3 = currentTour3;
        } else {
            // free(currentTour[i]);
        }
    }
    // printf("Here3\n");
       struct {
        double cost;
        int rank;
    } localData1 = {minTourCost1,world_rank},localData2 = {minTourCost2,world_rank}, localData3 = {minTourCost3,world_rank},globalData1,globalData2,globalData3;

    MPI_Reduce(&localData1, &globalData1, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
    MPI_Bcast(&globalData1.rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(bestTour1, numOfCoords + 1, MPI_INT, globalData1.rank, MPI_COMM_WORLD);

    MPI_Reduce(&localData2, &globalData2, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
    MPI_Bcast(&globalData2.rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(bestTour2, numOfCoords + 1, MPI_INT, globalData2.rank, MPI_COMM_WORLD);

    MPI_Reduce(&localData3, &globalData3, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
    MPI_Bcast(&globalData3.rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(bestTour3, numOfCoords + 1, MPI_INT, globalData3.rank, MPI_COMM_WORLD);

    // MPI_Bcast(bestTour, numOfCoords + 1, MPI_INT, globalData.rank, MPI_COMM_WORLD);
    if(world_rank == 0){
        double tEnd = omp_get_wtime();
        printf("\nTook %f milliseconds", (tEnd - tStart) * 1000);
        writeTourToFile(bestTour2, numOfCoords + 1, outFileName1);
        writeTourToFile(bestTour1, numOfCoords + 1, outFileName2);
        writeTourToFile(bestTour3, numOfCoords + 1, outFileName3);
    }
	//Free memory

    MPI_Finalize();

    // for(int i = 0; i < numOfCoords; i++){
    //     free(dMatrix[i]);
    // }
    
    // free(dMatrix);
    // for(int i = 0;i < 3;i++){
    //     free(bestTour[i]);
    // }
    return 0;
}

int *nearestAddition(double **dMatrix, int numOfCoords,int startVertex){
    // Initialize variables
    int nextNode = -1, insertPos = -1;

    // Allocate memory for the tour
    int *tour = (int *)malloc((numOfCoords + 1) * sizeof(int));
    bool *visited = (bool *)calloc(numOfCoords, sizeof(bool));
    for(int i = 0; i < numOfCoords; i++) {
        tour[i] = -1;
    }

  	// Initialize the tour with the start vertex
    tour[0] = startVertex;
    tour[1] = startVertex;
    visited[startVertex] = true;

    int numVisited = 1;
    int tourLength = 2;

    // Setup for parallelism
    int numThreads = omp_get_max_threads();
	// numThreads = 1;
    double *threadMinCosts = (double*)malloc(numThreads * 8 * sizeof(double));
    int *threadNextNode = (int*)malloc(numThreads * 8 * sizeof(int));
    int *threadInsertPos = (int*)malloc(numThreads * 8 * sizeof(int));

    #pragma omp parallel
    {
        int threadID = omp_get_thread_num();
		// threadID = 0;
        while(numVisited < numOfCoords){
            threadMinCosts[threadID * 8] = __DBL_MAX__;
            threadNextNode[threadID * 8] = -1;
            threadInsertPos[threadID * 8] = -1;

            // Find the nearest unvisited vertex
            #pragma omp for nowait
            for(int j = 0; j < numOfCoords; j++){
                if(!visited[j]){
                    for(int i = 0; i < tourLength - 1; i++){
                        double cost = dMatrix[tour[i]][j];
                        if(cost < threadMinCosts[threadID * 8]){
                            threadMinCosts[threadID * 8] = cost;
                            threadNextNode[threadID * 8] = j;
                            threadInsertPos[threadID * 8] = i;
                        }
                    }
                }
            }
			
            // Synchronization point
            #pragma omp barrier

            // Choose the nearest vertex among all threads
            #pragma omp single
            {
                double minCost = __DBL_MAX__;
				// printf("numThreads: %d\n", numThreads);
                for(int i = 0; i < numThreads; i++){
                    if(threadMinCosts[i * 8] < minCost){
                        minCost = threadMinCosts[i * 8];
                        nextNode = threadNextNode[i * 8];
                        insertPos = threadInsertPos[i * 8];
                    }
                }
                if(insertPos == 0){
                    // printf("Inserting %d before %d\n", nextNode, tour[insertPos]);
                    double afterstart = dMatrix[tour[0]][nextNode] + dMatrix[nextNode][tour[1]] - dMatrix[tour[0]][tour[1]];
                    double beforeend = dMatrix[tour[tourLength-2]][nextNode] + dMatrix[nextNode][tour[tourLength-1]] - dMatrix[tour[tourLength - 2]][tour[tourLength - 1]];
                    if(afterstart <= beforeend){
                        insertPos = 0;
                    } else {
                        insertPos = tourLength - 1;
                    }
                }
                
                // Decide where to insert nextNode (before or after the insertPos)
                double costBefore = (insertPos > 0) ? dMatrix[tour[insertPos - 1]][nextNode] + dMatrix[nextNode][tour[insertPos]] - dMatrix[tour[insertPos - 1]][tour[insertPos]] : __DBL_MAX__;
                double costAfter = (insertPos < tourLength - 1) ? dMatrix[tour[insertPos]][nextNode] + dMatrix[nextNode][tour[insertPos + 1]] - dMatrix[tour[insertPos]][tour[insertPos + 1]] : __DBL_MAX__;

                if (costBefore <= costAfter) {
                    // Insert before insertPos
                    for(int i = numOfCoords; i >= insertPos; i--){
                        tour[i + 1] = tour[i];
                    }
                    tour[insertPos] = nextNode;
                } else {
                    // Insert after insertPos
                    for(int i = numOfCoords; i > insertPos; i--){
                        tour[i] = tour[i - 1];
                    }
                    tour[insertPos + 1] = nextNode;
                }

                visited[nextNode] = true;
                tourLength++;
                numVisited++;
            }
        }
		// #pragma omp barrier
    }
    // Free memory and return the tour
	// for(int i = 0; i < numOfCoords; i++){
	// 	printf("%d\n", visited[i]);
	// }

	/////////////??
	// free(visited);
    //////////////
	// free(threadMinCosts);
    // free(threadNextNode);
    // free(threadInsertPos);
    return tour;
}

int *cheapestInsertion(double **dMatrix, int numOfCoords, int startVertex){

	//Setting up variables
	int nextNode, insertPos;

	//Memory allocation for the tour and visited arrays. Tour is numOfCoords + 1 for returning to origin
	//Visited uses calloc, array is instantiated with "0" as all elements. Good for boolean arrays.
	int *tour = (int *)malloc((1 + numOfCoords) * sizeof(int));
	bool *visited = (bool *)calloc(numOfCoords, sizeof(bool));

	if(tour == NULL){
		printf("Memory allocation failed");
		exit(EXIT_FAILURE);
	}

	//Initialising tour to empty
	for(int i = 0; i < numOfCoords; i++){
		tour[i] = -1;
	}

	// Initialize the tour with the start vertex
    tour[0] = startVertex;
    tour[1] = startVertex;
    visited[startVertex] = true;
	
	//Hard coding because I'm lazy
	int numVisited = 1;
	int tourLength = 2;

	//Where OMP starts... Get the env variable for the max num of threads.
	int numThreads = omp_get_max_threads();

	// printf("This program uses %d threads \n\n", numThreads);
	
	/*
	Set up arrays to be the size of the number of threads. Each thread will store 
	its minCost, its nextNode, and its insertPos in its respective memory location.
	Thread 0 will store its results at position 0, thread 1 will store its results at position 1 etc.
	*/

	double *threadMinCosts = NULL;
	int *threadNextNode = NULL;
	int *threadInsertPos = NULL;
		
	threadMinCosts = (double*)malloc(numThreads * 8 * sizeof(double));
	threadNextNode = (int*)malloc(numThreads * 8 * sizeof(int));
	threadInsertPos = (int*)malloc(numThreads * 8 * sizeof(int));

	//Start a parallel section
	#pragma omp parallel 
	{
		//Each thread now has started, and it stores its thread number in threadID
		int threadID = omp_get_thread_num();
		while(numVisited < numOfCoords){

			//Thread only accesses its memory location in the shared array. No race conditions.
			threadMinCosts[threadID * 8] = __DBL_MAX__;
			threadNextNode[threadID * 8] = -1;
			threadInsertPos[threadID * 8] = -1;

			//Begin a workshare construct. Threads divide i and j and work on their respective iterations.
			#pragma omp for collapse(2)
			for(int i = 0; i < tourLength - 1; i++){	
				for(int j = 0; j < numOfCoords; j++){

					//Each thread performs their cheapest insertion. Works on each position in the tour.
					if(!visited[j]){
						double cost = dMatrix[tour[i]][j] + dMatrix[tour[i+1]][j] - dMatrix[tour[i]][tour[i + 1]];
						if(cost < threadMinCosts[threadID * 8]){

							threadMinCosts[threadID * 8] = cost;
							threadNextNode[threadID * 8] = j;
							threadInsertPos[threadID * 8] = i + 1;
						}
					}
				}
			}

			//Only one thread works on this part. This part must be serial. OMP single instead of master. Therefore implicit barrier
			#pragma omp single
			{
				int bestNextNode = -1;
				int bestInsertPos = -1;
				double minCost = __DBL_MAX__;

				//A single thread loops through each threads memory locations. Finds the minCost
				for(int i = 0; i < numThreads; i++){
					if(threadMinCosts[i * 8] < minCost){
						minCost = threadMinCosts[i * 8];
						bestNextNode = threadNextNode[i * 8];
						bestInsertPos = threadInsertPos[i * 8];
					}
				}	

				//One thread places the bestNextNode in the bestInsertPos
				for(int i = numOfCoords; i > bestInsertPos; i--){
					tour[i] = tour[i - 1];
				}

				tour[bestInsertPos] = bestNextNode;
				visited[bestNextNode] = true;		
				
				tourLength++;
				numVisited++;

			}
		}
	}

	//Free all memory when done
	
	// free(visited);
	// free(threadMinCosts);
	// free(threadNextNode);
	// free(threadInsertPos);

	return tour;
}

int *farthestInsertion(double **dMatrix, int numOfCoords, int startVertex){
	//Setting up variables
	int nextNode, insertPos;

	//Memory allocation for the tour and visited arrays. Tour is numOfCoords + 1 for returning to origin
	//Visited uses calloc, array is instantiated with "0" as all elements. Good for boolean arrays.
	int *tour = (int *)malloc((1 + numOfCoords) * sizeof(int));
	bool *visited = (bool *)calloc(numOfCoords, sizeof(bool));

	//Initialising tour to empty
	for(int i = 0; i < numOfCoords; i++){
		tour[i] = -1;
	}

	// Initialize the tour with the start vertex
    tour[0] = startVertex;
    tour[1] = startVertex;
    visited[startVertex] = true;

	//Hard coding because I'm lazy
	int numVisited = 1;
	int tourLength = 2;

	//Where OMP starts... Get the env variable for the max num of threads.
	int numThreads = omp_get_max_threads();
	
	/*
	Set up arrays to be the size of the number of threads. Each thread will store 
	its minCost, its nextNode, and its insertPos in its respective memory location.
	Thread 0 will store its results at position 0, thread 1 will store its results at position 1 etc.
	Multiply by 8 to avoid false sharing. Each space is 64 bytes long (to ensure each thread has its own cache line)
	*/

	double *threadMinCosts = NULL;
	double *threadMaxCosts = NULL;
	int *threadNextNode = NULL;
	int *threadInsertPos = NULL;
		
	threadMinCosts = (double*)malloc(numThreads * 8 * sizeof(double));
	threadMaxCosts = (double*)malloc(numThreads * 8 * sizeof(double));
	threadNextNode = (int*)malloc(numThreads * 8 *sizeof(int));
	threadInsertPos = (int*)malloc(numThreads * 8 *sizeof(int));
	
	int bestNextNode = -1;

	//Start a parallel section
	#pragma omp parallel 
	{

	//Each thread now has started, and it stores its thread number in threadID
	int threadID = omp_get_thread_num();
	
	while(numVisited < numOfCoords){

		//Point 1: Thread only accesses its memory location in the shared array.
		threadMinCosts[threadID * 8] = __DBL_MAX__;
		threadMaxCosts[threadID * 8] = 0;
		threadNextNode[threadID * 8] = -1;
		threadInsertPos[threadID * 8] = -1;

		//Begin a workshare construct. Threads divide i and j and work on their respective ones.
		#pragma omp for collapse(2)
		for(int i = 0; i < tourLength - 1; i++){
			for(int j = 0; j < numOfCoords; j++){
				//Each thread identifies their farthest vertex from a vertex in the tour
				if(!visited[j]){
					double cost = dMatrix[tour[i]][j];
					if(cost > threadMaxCosts[threadID * 8]){
						//See Point 1
						threadMaxCosts[threadID * 8] = cost;
						threadNextNode[threadID * 8] = j;
					}
				}
			}
		}

		//Single construct, one thread looks through what each thread has. Choosest the farthest node.
		#pragma omp single
		{
			double maxCost = 0;
			for(int i = 0; i < numThreads; i++){
				if(threadMaxCosts[i * 8] > maxCost){
					maxCost = threadMaxCosts[i * 8];
					bestNextNode = threadNextNode[i * 8];
				}
			}

			
		}

		//Find the cost of adding the farthest node to every possible location in the tour. Each thread finds their own.
		#pragma omp for
		for(int k = 0; k < tourLength - 1; k++){
			double cost = dMatrix[tour[k]][bestNextNode] + dMatrix[bestNextNode][tour[k + 1]] - dMatrix[tour[k]][tour[k + 1]];
			if(cost < threadMinCosts[threadID * 8]){
				threadMinCosts[threadID * 8] = cost;
				threadInsertPos[threadID * 8] = k + 1;
			}
		}

		//Single construct only one thread working on this part.
		#pragma omp single
		{
		int bestInsertPos = -1;
		double minCost = __DBL_MAX__;

		//Single thread loops through every thread's answer and chooses the cheapest one.
		for(int i = 0; i < numThreads; i++){
			if(threadMinCosts[i * 8] < minCost){
				minCost = threadMinCosts[i * 8];
				bestInsertPos = threadInsertPos[i * 8];
			}
		}	

		//Single thread places the bestNextNode in the bestInsertPos
		for(int i = numOfCoords; i > bestInsertPos; i--){
			tour[i] = tour[i - 1];
		}

		tour[bestInsertPos] = bestNextNode;
		visited[bestNextNode] = true;		
		
		tourLength++;
		numVisited++;

		}
	}
	}

	//Free all memory when done
	

	// free(visited);
	// free(threadMinCosts);
	// free(threadNextNode);
	// free(threadInsertPos);
	// free(threadMaxCosts);

	return tour;
}


double **createDistanceMatrix(double **coords, int numOfCoords){
	int i, j;
	
	double **dMatrix = (double **)malloc(numOfCoords * sizeof(double));

	for(i = 0; i < numOfCoords; i++){
		dMatrix[i] = (double *)malloc(numOfCoords * sizeof(double));
	}

	#pragma omp parallel for collapse(2)
	for(i = 0; i < numOfCoords; i++){
		for(j = 0; j < numOfCoords; j++){
			double diffX = coords[i][0] - coords[j][0];
			double diffY = coords[i][1] - coords[j][1];
			dMatrix[i][j] = sqrt((diffX * diffX) + (diffY * diffY));
		}
	}

	return dMatrix;
}
