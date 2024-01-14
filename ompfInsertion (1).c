#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<stdbool.h>
#include<math.h>
#include<omp.h>


int readNumOfCoords(char *filename);
double **readCoords(char *filename, int numOfCoords);
void *writeTourToFile(int *tour, int tourLength, char *filename);
double **createDistanceMatrix(double **coords, int numOfCoords);
double sqrt(double arg);
int *farthestInsertion(double **dMatrix, int numOfCoords,int startVertex);

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
        int *currentTour = farthestInsertion(dMatrix, numOfCoords, v);
        
        // Calculate the cost of the current tour
        double currentTourCost = 0;
        for(int i = 0; i < numOfCoords; i++) {
            currentTourCost += dMatrix[currentTour[i]][currentTour[i+1]];
        }
        currentTourCost = currentTourCost + v*0.0000001;

        // Check if the current tour is the cheapest
        if(currentTourCost < minTourCost) {
            minTourCost = currentTourCost;
            free(bestTour); // Free the previous best tour
            bestTour = currentTour;
        } else {
            free(currentTour);
        }
    }

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
	

	free(visited);
	free(threadMinCosts);
	free(threadNextNode);
	free(threadInsertPos);
	free(threadMaxCosts);

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
