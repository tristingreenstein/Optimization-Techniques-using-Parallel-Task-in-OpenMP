// C program to implement Runge
// Kutta method
  
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h> 
// A sample differential equation
// "dy/dx = (x - y)/2"
double dydx(double x, double y)
{
    return (x + y -2);
}
  
// Finds value of y for a given x
// using step size h
// and initial value y0 at x0.
double rungeKutta(double x0, double y0, double x, double h) {
    // Count number of iterations
    // using step size or
    // step height h
    int n = (int)((x - x0) / h);
  
    double k1, k2;
  
    // Iterate for number of iterations
    double y = y0;
   
    for (int i = 1; i <= n; i++) {
        // Apply Runge Kutta Formulas
        // to find next value of y
        k1 = h * dydx(x0, y);
      
        k2 = h * dydx(x0 + 0.5 * h,
                      y + 0.5 * k1);
  
        // Update next value of y
        y = y + (1.0 / 6.0) * (k1 + 2 * k2);
  
        // Update next value of x
        x0 = x0 + h;
    }
  
    return y;
}
  

double rungeKuttaParallel(double x0, double y0, double x, double h) {
    // Count number of iterations
    // using step size or
    // step height h
    int n = (int)((x - x0) / h);
  
    double k1, k2;
  
    // Iterate for number of iterations
    double y = y0;
   #pragma omp parallel master 
   {
    for (int i = 1; i <= n; i++) {
        // Apply Runge Kutta Formulas
        // to find next value of y
      
       #pragma omp task depend(out: k1)
        {
        k1 = h * dydx(x0, y);
        }
        
      #pragma omp task depend(inout : k1)
       {
        k2 = h * dydx(x0 + 0.5 * h,
                      y + 0.5 * k1);
       }
    
       #pragma omp task depend(in : k1,k2)
        // Update next value of y
       {
        y = y + (1.0 / 6.0) * (k1 + 2 * k2);
       }
        #pragma omp taskwait  
        // Update next value of x
        x0 = x0 + h;
    }
  
   }
    
    return y;
} 
// Driver Code
int main()
{
    double x0 = 20, y = 100,
          x = 20, h =0.2450;
    double timeSpentParallel, timeSpentSequential;
    clock_t begin = clock();
    printf("y(x) = %lf\n",
           rungeKutta(x0, y, x, h));
     clock_t end = clock();
        timeSpentSequential= (double)(end - begin) / CLOCKS_PER_SEC;

     clock_t begin1 = clock();
    printf("y(x) (Parallel) = %lf\n",
           rungeKuttaParallel(x0, y, x, h));
           clock_t end1 = clock();
        timeSpentParallel= (double)(end1 - begin1) / CLOCKS_PER_SEC;

        printf("TimeSpentParallel: %lf \n TimeSpentSequential: %lf", timeSpentParallel, timeSpentSequential);
    return 0;
}