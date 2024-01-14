#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<math.h>
#include<string.h>
#include<omp.h>

// ... [Other functions: readNumOfCoords, readCoords, writeTourToFile, createDistanceMatrix]

int readNumOfCoords(char *filename);
double **readCoords(char *filename, int numOfCoords);
void *writeTourToFile(int *tour, int tourLength, char *filename);
double **createDistanceMatrix(double **coords, int numOfCoords);
int *nearestAddition(double **dMatrix, int numOfCoords,int startVertex);

int main(int argc, char *argv[]){
    
	if(argc != 3){
		printf("Program should be called as ./program <coordFile> <outFileName>");
		return 1;
	}


	//Argument setup for file and output
	char filename[500];
	char outFileName[500];

	strcpy(filename, argv[1]);
	strcpy(outFileName, argv[2]);

	//Reading files and setting up the distance matrix
	int numOfCoords = readNumOfCoords(filename);
	double **coords = readCoords(filename, numOfCoords);

	double tStart = omp_get_wtime();

	/*Program starts*/

    double **dMatrix = createDistanceMatrix(coords, numOfCoords);
    
	double minTourCost = __DBL_MAX__;
    int *bestTour = NULL;

    for(int v = 0; v < numOfCoords; v++) {
		
        int *currentTour = nearestAddition(dMatrix, numOfCoords, v);
        
        // Calculate the cost of the current tour
        double currentTourCost = 0;
        for(int i = 0; i < numOfCoords; i++) {
            currentTourCost += dMatrix[currentTour[i]][currentTour[i+1]];
        }
        currentTourCost = currentTourCost + v*0.0000001;

        printf("Tour cost for %d : %f\n", v,currentTourCost);
        // Check if the current tour is the cheapest
        if(currentTourCost < minTourCost) {
            minTourCost = currentTourCost;
            free(bestTour); // Free the previous best tour
            bestTour = currentTour;
        } else {
            free(currentTour);
        }
    }
	// bestTour = nearestAddition(dMatrix, numOfCoords, 0);
	// printf("Best tour:\n");
	// for(int i = 0; i < numOfCoords + 1; i++){
	// 	printf("%d\n", bestTour[i]);
	// }
    /*Program ends*/

	double tEnd = omp_get_wtime();

	printf("\nTook %f milliseconds", (tEnd - tStart) * 1000);

	if (writeTourToFile(bestTour, numOfCoords + 1, outFileName) == NULL){
		printf("Error");
	}

	//Free memory
	for(int i = 0; i < numOfCoords; i++){
		free(dMatrix[i]);
	}
	
	free(dMatrix);
	free(bestTour);
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
    double *threadMinCosts = (double*)malloc(numThreads * 8 * sizeof(double));
    int *threadNextNode = (int*)malloc(numThreads * 8 * sizeof(int));
    int *threadInsertPos = (int*)malloc(numThreads * 8 * sizeof(int));

    #pragma omp parallel
    {
        int threadID = omp_get_thread_num();

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
	free(threadMinCosts);
    free(threadNextNode);
    free(threadInsertPos);
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

