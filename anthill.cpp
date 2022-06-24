// #include <stdio.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <omp.h>
#include <utility>
using namespace std;

// ---------------------------------------------------------------------------------
// Define Lawn_Class
// Variables:
//  Lawn - Represents a lawn that has been divided into a 2D grid of cells (like
//  a
//         graphing sheet); Lawn is a 2D array where L[i][j] holds the expected
//         number of ants
//
class Lawn_Class {
public:
  // Lawn query functions
  double number_of_ants_in_cell(int, int);
  int guess_anthill_location(int, int);
  void report_results(double);
  // Lawn functions
  void initialize_Lawn(int, int, int, int);
  void update_Lawn();
  // Helper functions
  double sum_Lawn();
  void print_Lawn();
  void print_XY();
  void save_Lawn_to_file();

  int m; // Number of cells along each dimension of the Lawn

private:
  int n;         // Total number of cells = m x m
  int anthill_x; // Anthill is located at (anthill_x,anthill_y)
  int anthill_y; // in the Lawn
  int *X;        // Cell k is located at (X[k],Y[k])
  int *Y;
  double **Lawn;               // Stores number of ants per cell
  double **Lawn2;              // Work array used when updating Lawn
  int query_count;             // Number of queries to number_of_ants_in_cell
  int guess_count;             // Number of guesses for anthill location
  int steps;                   // Number of steps of simulation
  double penalty;              // Time used to penalize guesses and queries
  double time_to_find_anthill; // Time needed to find anthill
  // Lawn functions
  void set_anthill_location(int, int);
  void add_ant_to_Lawn();
  void swap_Lawn_with_Lawn2();
};
// ---------------------------------------------------------------------------------
// Returns the expected number of ants in cell (i,j) of the Lawn
//  - increments query_count
double Lawn_Class::number_of_ants_in_cell(int i, int j) {
#pragma omp atomic update
  query_count++;

  return Lawn[i][j];
}
// ---------------------------------------------------------------------------------
// Determines if the guess that the anthill is at cell (,j,) is true or false
//  - increments guess_count
int Lawn_Class::guess_anthill_location(int i, int j) {
#pragma omp atomic update
  guess_count++;

  if ((i == anthill_x) && (j == anthill_y)) {
    return 1;
  } else {
    return 0;
  }
}
// ---------------------------------------------------------------------------------
// Report results
void Lawn_Class::report_results(double execution_time) {

  printf("Size = %d x %d, Ants = %8.1f, Queries = %d, Guesses = %d, Penalty = "
         "%f, Execution time = %f\n",
         m, m, sum_Lawn(), query_count, guess_count, penalty, execution_time);
}
// ---------------------------------------------------------------------------------
// Initialize Lawn variables
// - size = number of cells along each dimenstion of the Lawn
void Lawn_Class::initialize_Lawn(int side, int anthill_x, int anthill_y,
                                 int nsteps) {

  // Initialize Lawn sizes
  m = side;  // Number of cells along each dimension of the Lawn
  n = m * m; // Total number of cells = m x m

  // Initialize X and Y arrays
  X = new int[n];
  Y = new int[n];
  int idx = 0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      X[idx] = i;
      Y[idx] = j;
      idx++;
    }
  }
  // Set up Lawn as a m x m 2D array from 1D array of length n = m*m
  Lawn = new double *[m];
  double *array = new double[n];
  for (int i = 0; i < m; i++)
    Lawn[i] = &(array[i * m]);

  // Set up Lawn2 similar to Lawn; to be used as a work array when updating Lawn
  Lawn2 = new double *[m];
  array = new double[n];
  for (int i = 0; i < m; i++)
    Lawn2[i] = &(array[i * m]);

  // Initialize Lawn cells with zero ants
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      Lawn[i][j] = 0.0;
    }
  }
  // Place anthill in the lawn at cell (anthill_x,anthill_y)
  set_anthill_location(anthill_x, anthill_y);

  // Update the lawn steps number of times
  steps = nsteps;
  double time_start = omp_get_wtime();
  for (int s = 0; s < steps; s++) {
    update_Lawn();
  }
  // Initialize penalty equal to update_Lawn time multiplied by steps
  penalty = omp_get_wtime() - time_start;

  time_to_find_anthill = 0.0; // Set time_to_find_anthill to zero
  query_count = 0;            // Set number of queries made to zero
  guess_count = 0;            // Set number of guesses made to zero
}
// ---------------------------------------------------------------------------------
// Update Lawn every time step by calculating expected number of ants in each
// cell.
// - Lawn is "renamed" Lawn2 so that Lawn2 has current expected ant values
// - Expected ant values for next time step will be computed from Lawn2
//   and placed in Lawn, as shown below:
//   Lawn[i][j] = 0.25*(Lawn2[i-1][j]+Lawn2[i+1][j]+Lawn2[i][j-1]+Lawn2[i][j+1])
// - Non-existent neighbors of cell (i,j) have zero contribution
void Lawn_Class::update_Lawn() {
  int i, j;
  // Rename Lawn as Lawn2, Lawn2 as Lawn
  swap_Lawn_with_Lawn2();

  // Update Lawn from Lawn2 values
  double time_start = omp_get_wtime();
#pragma omp parallel for
  for (i = 0; i < m; i++) {
    for (j = 0; j < m; j++) {
      Lawn[i][j] = 0.0;
      if (i > 0)
        Lawn[i][j] += 0.25 * Lawn2[i - 1][j];
      if (i < m - 1)
        Lawn[i][j] += 0.25 * Lawn2[i + 1][j];
      if (j > 0)
        Lawn[i][j] += 0.25 * Lawn2[i][j - 1];
      if (j < m - 1)
        Lawn[i][j] += 0.25 * Lawn2[i][j + 1];
    }
  }
  // Add ant to Lawn
  add_ant_to_Lawn();
}
// ---------------------------------------------------------------------------------
// Place anthill at cell (x,y) as part of initialization
void Lawn_Class::set_anthill_location(int x, int y) {
  anthill_x = x;
  anthill_y = y;
  if (x < 0)
    anthill_x = 0;
  else if (x >= m)
    anthill_x = m - 1;

  if (y < 0)
    anthill_y = 0;
  else if (y >= m)
    anthill_y = m - 1;
}
// ---------------------------------------------------------------------------------
// Add an ant to the anthill
void Lawn_Class::add_ant_to_Lawn() { Lawn[anthill_x][anthill_y] += 1.0; }
// ---------------------------------------------------------------------------------
// Swap Lawn with Lawn2 (Lawn2 is used as workspace)
void Lawn_Class::swap_Lawn_with_Lawn2() {
  double **temp = Lawn; // Replace Lawn 2 with Lawn to fix bug
  Lawn = Lawn2;
  Lawn2 = temp;
}
// ---------------------------------------------------------------------------------
// Add up all the values in Lawn, which gives the expected number of ants
// in the Lawn; after sufficient number of time steps when the anyt population
// reaches an equilibrium, this value does not change with time steps.
double Lawn_Class::sum_Lawn() {
  int i, j;
  double sum = 0.0;
  for (i = 0; i < m; i++) {
    for (j = 0; j < m; j++) {
      sum += Lawn[i][j];
    }
  }
  return sum;
}
// ---------------------------------------------------------------------------------
// Print Lawn, for debugging purposes
// - Lawn[i][j] is the expected number of ants in cell(i,j)
void Lawn_Class::print_Lawn() {
  int i, j;
  for (i = 0; i < m; i++) {
    printf("Row = %d: ", i);
    for (j = 0; j < m; j++) {
      printf("%8.4f", Lawn[i][j]);
    }
    printf("\n");
  }
}
// ---------------------------------------------------------------------------------
// Print X and Y array, for debugging purposes
// - Cell k is at (X[k],Y[k])
void Lawn_Class::print_XY() {
  int i, j;
  int idx = 0;
  for (i = 0; i < m; i++) {
    printf("Row = %d: ", i);
    for (j = 0; j < m; j++) {
      printf("(%d,%d) \t", X[idx], Y[idx]);
      idx++;
    }
    printf("\n");
  }
}
// ---------------------------------------------------------------------------------
// Save Lawn to file for visualization via MATLAB
void Lawn_Class::save_Lawn_to_file() {
  int i, j;
  int idx = 0;
  FILE *outfile;
  outfile = fopen("Lawn_Data", "w");
  for (i = 0; i < m; i++) {
    for (j = 0; j < m; j++) {
      fprintf(outfile, "%d %d %f\n", X[idx], Y[idx], Lawn[i][j]);
      idx++;
    }
  }
  fclose(outfile);
}
// ---------------------------------------------------------------------------------
// Main program to find the anthill
//
bool exist(int x, int y, Lawn_Class &);

int main(int argc, char **argv) {

  double start_time, execution_time; // Variables to determine execution time

  Lawn_Class MyLawn; // Create object MyLawn

  if (argc != 5) {
    printf(
        "Use: <executable_name> <lawn_size> <anthill_x> <anthill_y> <steps>\n");
    printf(" <lawn_size> = number of cells along each dimension of the lawn\n");
    printf(" <anthill_x> <anthill_y> = location of the anthill\n");
    printf(" <steps> = number of steps the lawn is updated\n");
    return 0;
  }
  int size = atoi(argv[1]);
  int anthill_x = atoi(argv[2]);
  int anthill_y = atoi(argv[3]);
  int steps = atoi(argv[4]);

  MyLawn.initialize_Lawn(size, anthill_x, anthill_y, steps);

  MyLawn.save_Lawn_to_file(); // Not recommended when size is large
                              /*
                               int n = MyLawn.m;
                               int m = MyLawn.m;
                               int lo = 0;
                               int hi = m-1;
                               int mid;
                               while(lo <= hi){
                                   mid = lo +( hi - lo) / 2;
                                   int max_row = 0;
                                   for(int i = 0; i < n; i++){
                                       if(MyLawn.number_of_ants_in_cell(max_row, mid) <
                               MyLawn.number_of_ants_in_cell(i, mid)) max_row = i;
                                   }
                            
                                   if((mid == 0 || MyLawn.number_of_ants_in_cell(max_row, mid)) >
                               MyLawn.number_of_ants_in_cell(max_row, mid-1) && (mid == m-1 ||
                               MyLawn.number_of_ants_in_cell(max_row, mid) >
                               MyLawn.number_of_ants_in_cell(max_row, mid-1))){
                                       if(MyLawn.guess_anthill_location(max_row, mid) == 1){
                                           execution_time = omp_get_wtime() - start_time;
                            
                                           MyLawn.report_results(execution_time);
                               // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                           return 0;
                                       }
                                   }else if(mid > 0 && MyLawn.number_of_ants_in_cell(max_row, mid-1) >
                               MyLawn.number_of_ants_in_cell(max_row, mid)) hi = mid-1;                             else{                             lo = mid+1;
                                   }
                               }
                           */

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Example Approach #1: Brute force approach
  // double res = 1000000000;
#pragma omp declare reduction (maximum:pair<double, pair<int, int>>: omp_out = omp_out.first > omp_in.first ? omp_out : omp_in)
  pair<double, pair<int, int>> res = make_pair(-1, make_pair(-1, -1));
  start_time = omp_get_wtime();
  volatile int found = 0;
#pragma omp parallel for default(none) shared(MyLawn, found) reduction(maximum    \
                                                                        : res)
  for (int i = 0; i < MyLawn.m; i++) {
    for (int j = 0; j < MyLawn.m; j++) {
      double current = MyLawn.number_of_ants_in_cell(i, j);
      if ((!exist(i - 1, j, MyLawn) ||
           current > MyLawn.number_of_ants_in_cell(i - 1, j)) &&
          (!exist(i, j - 1, MyLawn) ||
           current > MyLawn.number_of_ants_in_cell(i, j - 1)) &&
          (!exist(i + 1, j, MyLawn) ||
           current > MyLawn.number_of_ants_in_cell(i + 1, j)) &&
          (!exist(i, j + 1, MyLawn) ||
           current > MyLawn.number_of_ants_in_cell(i, j + 1))) {
        res = max(res, make_pair(current, make_pair(i, j)));
      }
    }
  }

  printf("%d, %d", res.second.first, res.second.second);
  /*
              if(i == 0){
                  if(j == 0){
                      if(current > MyLawn.number_of_ants_in_cell(1,0) && current
  > MyLawn.number_of_ants_in_cell(0, 1)){} }else if(j == MyLawn.m-1){ if(current
  > MyLawn.number_of_ants_in_cell(1,j) && current >
  MyLawn.number_of_ants_in_cell(0, j-1)){} }else{ if(current >
  MyLawn.number_of_ants_in_cell(0,j-1) && current >
  MyLawn.number_of_ants_in_cell(0, j+1) && current >
  MyLawn.number_of_ants_in_cell(1, j))
                  }
              }
                  else if(i == 0 && j == MyLawn.m-1){
                  if(current > MyLawn.number_of_ants_in_cell(1,j) && current >
  MyLawn.number_of_ants_in_cell(0, j-1)) }else if(i == MyLawn.m-1 && j == 0){
                  if(current > MyLawn.number_of_ants_in_cell(i-1,0) && current >
  MyLawn.number_of_ants_in_cell(i,j+1)) }else if(i == MyLawn.m-1 && j ==
  MyLawn.m-1){ if(current > MyLawn.number_of_ants_in_cell(i-1,j) && current >
  MyLawn.number_of_ants_in_cell(i, j-1)) }else{

              }
              if(MyLawn.number_of_ants_in_cell(i, j) :
              if (found == 0) {
                  if (MyLawn.guess_anthill_location(i,j) == 1) {
                      found = 1;
  #pragma omp flush(found)
                  }
              }
          }
    }*/
  // #pragma parallel for ends here ...
  execution_time = omp_get_wtime() - start_time;

  MyLawn.report_results(execution_time);
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return 0;
}

bool exist(int x, int y, Lawn_Class &MyLawn) {
  return 0 <= x && x < MyLawn.m && 0 <= y && y < MyLawn.m;
}