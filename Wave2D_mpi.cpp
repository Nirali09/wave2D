#include <iostream>
#include "Timer.h"
#include <stdlib.h>   // atoi
#include <math.h>
#include "mpi.h" //mpi
#include <omp.h> //openMP

int default_size = 100;  // the default system size
int defaultCellWidth = 8;
double c = 1.0;      // wave speed
double dt = 0.1;     // time quantum
double dd = 2.0;     // change in system
int t = 0;
int my_rank = 0;    //MPI processor ranks
int mpi_size = 1;   //#processors
/*double value1 = pow(c, 2 ) / 2; //part of formula 
double value2 = pow((dt/dd), 2); //part of formula
double constant = (value1 * value2); //usinng in the formula
*/
using namespace std;

int main( int argc, char *argv[] ) {
  // verify arguments
  if ( argc != 5 ) {
    cerr << "usage: Wave2D size max_time interval numThreads" << endl;
    return -1;
  }
  int size = atoi( argv[1] );
  int max_time = atoi( argv[2] );
  int interval  = atoi( argv[3] );
  int thread_num = atoi(argv[4]); //taking number of threads from user

  if ( size < 100 || max_time < 3 || interval < 0 || thread_num < 0) {
    cerr << "usage: Wave2D size max_time interval numThreads" << endl;
    cerr << "where size >= 100 && time >= 3 && interval >= 0 && numThreads >= 1" << endl;
    return -1;
  }

  //for MPI
  MPI_Init( &argc, &argv ); // start MPI
  MPI_Comm_rank( MPI_COMM_WORLD, &my_rank ); //get the rank
  MPI_Comm_size( MPI_COMM_WORLD, &mpi_size ); //get the number of processes
  MPI_Request request; //capture request of a MPI_Isend
  MPI_Status status; // store status of a MPI_Recv

  // create a simulation space 
  double z[3][size][size];
  for (int p = 0; p < 3; p++) 
    for (int i = 0; i < size; i++)
      for (int j = 0; j < size; j++)
	      z[p][i][j] = 0.0; // no wave

  int r = size % mpi_size; //remainder
  int num_rows = (size / mpi_size) + (my_rank < r ? 1 : 0); // partitioned stripe with added remainder
  int first = my_rank * (size / mpi_size) + (my_rank <= r ? my_rank : r); //first of stripe
  int last = first + num_rows;//last of the stripe

  //print each rank's range
  cerr <<"Rank[" << my_rank << "]'s range = " << first << " ~ " << last - 1 << endl;

  // change # of threads
  omp_set_num_threads( thread_num );

  // start a timer
  Timer time;
  if (my_rank == 0) {
    time.start( );
  }

  // time = 0;
  // initialize the simulation space: calculate z[0][][]
  int weight = size / default_size;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (i > 40 * weight && i < 60 * weight  && j > 40 * weight && j < 60 * weight) {
        z[0][i][j] = 20.0;
      } else {
          z[0][i][j] = 0.0;
      }
    }
  }
  
  // time = 1
  // calculate z[1][][]
  // cells not on edge
  // IMPLEMENT BY YOURSELF !!!
  for (int i = 1; i < size-1; i++) {
    for (int j = 1; j < size-1; j++) {
        // Zt_i,j = Zt-1_i,j + c2 / 2 * (dt/dd)2 * (Zt-1_i+1,j + Zt-1_i-1,j + Zt-1_i,j+1 + Zt-1_i,j-1 – 4.0 * Zt-1_i,j)
        z[1][i][j] = z[0][i][j] + (1.0/800.0) * (z[0][i + 1][j] + z[0][i - 1][j] + z[0][i][j + 1] + z[0][i][j - 1] - 4.0 * z[0][i][j]); //formula for wave at t=1
    }
  }

  // simulate wave diffusion from time = 2
  for (t = 2; t < max_time; t++) { //for t=2 to max_time-1 sharing boundry data

    if (t > 2) { // sending data in forward direction i.e. rank0->rank1
      if (my_rank < mpi_size - 1) { //for 0,1,2
        MPI_Isend( &z[(t - 1) % 3][0][0] + (size * (last - 1)), size, MPI_DOUBLE, my_rank + 1, 1, MPI_COMM_WORLD, &request);
      }

      if (my_rank > 0) { //sending data in backward direction i.e. rank2->rank1
        MPI_Isend( &z[(t - 1) % 3][0][0] + (size * first), size, MPI_DOUBLE, my_rank - 1, 1, MPI_COMM_WORLD, &request);
      }

      if (my_rank < mpi_size - 1){ //receiving data from backward direction
        MPI_Recv( &z[(t - 1) % 3][0][0] + (size * last) , size, MPI_DOUBLE, my_rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      }

      if (my_rank > 0){ //receiving data from forward direction
        MPI_Recv( &z[(t - 1) % 3][0][0] + (size * (first - 1)) , size, MPI_DOUBLE, my_rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      }
    }

    //calculate from t=2 
    //parallelizing using OpenMP
    #pragma omp parallel for
    for (int i = first; i < last; i++) {
      for (int j = 0; j < size; j++){
        if (i == 0 || j == 0 || i >= size - 1 || j >= size - 1 ) { //do not calculate when on end boundry of matrix 
            continue;
        }
        //using the "t % 3" to rotate matrix 
        //Zt_i,j = 2.0 * Zt-1_i,j – Zt-2_i,j + c2 * (dt/dd)2 * (Zt-1_i+1,j + Zt-1_i-1,j + Zt-1_i,j+1 + Zt-1_i,j-1 – 4.0 * Zt-1_i,j)
        z[t % 3][i][j] = (2.0 * z[(t - 1) % 3][i][j] - z[(t - 2) % 3][i][j] + pow(c, 2) * pow((dt/dd), 2) * (z[(t - 1) % 3][i+1][j] + z[(t - 1) % 3][i-1][j] + z[(t - 1) % 3][i][j+1] + z[(t - 1) % 3][i][j-1]- 4.0 * z[(t - 1) % 3][i][j]));
        // limiting water surface
        if(z[t % 3][i][j] > 20.0) { 
          z[t % 3][i][j] = 20.0;
        } else if(z[t % 3][i][j] < -20.0){
          z[t % 3][i][j] = -20.0;
        }
      }
    }

    if ((interval != 0 && t % interval == 0) || t == max_time - 1) {
      if(my_rank > 0) {
        MPI_Isend( &z[t % 3][0][0] + (size * first), size * num_rows, MPI_DOUBLE, 0, my_rank, MPI_COMM_WORLD, &request); // send from all slaves after calculation
      }
      //rank0 receiving data from all other workers
      if (my_rank == 0) {
        for (int j = 1; j < mpi_size; j++){
          int num_rows_j = (size / mpi_size) + (j < r ? 1 : 0); // num rows for worker j
          int first_j = j * (size / mpi_size) + (j <= r ? j : r); // first index for worker j
          MPI_Recv( &z[t % 3][0][0] + (size * first_j) , size * num_rows_j, MPI_DOUBLE, j, j, MPI_COMM_WORLD, &status);//receiving data to rank0
        }
        //printing data 
        if (interval != 0) {
          cout << t << endl;
          for (int m = 0; m < size; m++){
            for (int n = 0; n < size; n++){
              std::cout << z[t % 3][m][n] << " ";
            }
            std::cout << endl;
          }
          std::cout << endl;
        }
      }
    }
  }

  // end of simulation
  // finish the timer
  if (my_rank == 0) {
    cerr << "Elapsed time = " << time.lap( ) << endl;
  }
  
  MPI_Finalize( ); // shut down MPI

  return 0;
}



