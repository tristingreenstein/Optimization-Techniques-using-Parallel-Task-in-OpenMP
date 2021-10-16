#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
typedef double (*pfunc)(double, double, double, double (*)());
typedef double (*rfunc)(double);
 
#define INTG(F,A,B) (F((B))-F((A)))

double intlLeftRectParallel(double from, double to, double n, double (*func)()) {
    
  double h = (to-from)/n;
  double sum = 0.0, x;
  #pragma omp parallel private(x)
  #pragma omp master 
  {
    x = from;
    while (x <= (to-h)){
    
      #pragma omp task firstprivate(x) 
      {
        sum += func(x);
      }
      #pragma omp taskwait
      x += h;
    }  
  }
  return h*sum;  
}

double int_leftrect(double from, double to, double n, double (*func)())
{
   double h = (to-from)/n;
   double sum = 0.0, x;
   for(x=from; x <= (to-h); x += h)
      sum += func(x);
   return h*sum;
}

double intRightrectParallel(double from, double to, double n, double (*func)())
{
   double h = (to-from)/n;
   double sum = 0.0, x;
   #pragma omp parallel private(x)
   {
     #pragma omp master
     {
      for(x=from; x <= (to-h); x += h){
        #pragma omp task firstprivate(x)
        {
        sum += func(x+h);
        }
        #pragma omp taskwait
      }
     }
   }
    return h*sum;
}
 
double int_rightrect(double from, double to, double n, double (*func)())
{
   double h = (to-from)/n;
   double sum = 0.0, x;
   for(x=from; x <= (to-h); x += h)
     sum += func(x+h);
   
   return h*sum;
}
 
 double intMidrectParallel(double from, double to, double n, double (*func)())
{
   double h = (to-from)/n;
   double sum = 0.0, x;
   #pragma omp parallel private(x)
   {
     #pragma omp master
     {
      for(x=from; x <= (to-h); x += h){
        #pragma omp task firstprivate(x)
        {
          sum += func(x+h/2.0);
        }
        #pragma omp taskwait
      }
     }
   }
   return h*sum;
}

double int_midrect(double from, double to, double n, double (*func)())
{
   double h = (to-from)/n;
   double sum = 0.0, x;
   for(x=from; x <= (to-h); x += h)
     sum += func(x+h/2.0);
   return h*sum;
}

double intTrapeziumParallel(double from, double to, double n, double (*func)())
{
   double h = (to - from) / n;
   double sum = func(from) + func(to);
   int i;  
   #pragma omp parallel for private(i,from) reduction(+:sum) shared(h,n)
      for(i = 1; i < ((int)n);i++){
        sum += 2.0*func(from + i * h);
      }
   return  h * sum / 2.0;
}

double int_trapezium(double from, double to, double n, double (*func)())
{
   double h = (to - from) / n;
   double sum = func(from) + func(to);
   int i;
   for(i = 1;i < n;i++)
       sum += 2.0*func(from + i * h);
   return  h * sum / 2.0;
}

double intSimpsonParallel(double from, double to, double n, double (*func)())
{
   double h = (to - from) / n;
   double sum1 = 0.0;
   double sum2 = 0.0;
   int i;
 
   double x;
   #pragma omp parallel for private(i) reduction(+ : sum1) shared(from,h) 
   for(i = 0;i < ((int)n);i++)
      sum1 += func(from + h * i + h / 2.0);
 
   #pragma omp parallel for private(i) reduction(+ : sum2) shared(from,h) 
   for(i = 1;i < ((int)n);i++)
      sum2 += func(from + h * i);
 
   return h / 6.0 * (func(from) + func(to) + 4.0 * sum1 + 2.0 * sum2);
}

double int_simpson(double from, double to, double n, double (*func)())
{
   double h = (to - from) / n;
   double sum1 = 0.0;
   double sum2 = 0.0;
   int i;
 
   double x;
 
   for(i = 0;i < n;i++)
      sum1 += func(from + h * i + h / 2.0);
 
   for(i = 1;i < n;i++)
      sum2 += func(from + h * i);
 
   return h / 6.0 * (func(from) + func(to) + 4.0 * sum1 + 2.0 * sum2);
}
/* test */
double f3(double x)
{
  return x;
}
 
double f3a(double x)
{
  return x*x/2.0;
}
 
double f2(double x)
{
  return 1.0/x;
}
 
double f2a(double x)
{
  return log(x);
}
 
double f1(double x)
{
  return x*x*x;
}
 
double f1a(double x)
{
  return x*x*x*x/4.0;
}
void checkDifference(double parallel, double sequential){
if(parallel > sequential){
  printf("sequential  method took %f  and is faster than parallel by: %f\n", sequential, (parallel-sequential));
}else {
  printf("Parallel method took %f and is faster than sequential by: %f\n" ,parallel ,(sequential-parallel));
}

}
void iteratesThroughTwoFunctions(rfunc rf[],rfunc If[],double ivals[],double approx[], pfunc f[], const char *names[2]){
  double ic;
  double timeSpentParallel,timeSpentSequential;
  int i, j;
  clock_t begin = clock();
    for(j=0; j < 4; j++)
      {
        for(i=0; i < 1; i++)
        {
          ic = (*f[i])(ivals[2*j], ivals[2*j+1], approx[j], rf[j]);
          printf("%10s [ 0,1] num: %+lf, an: %lf\n",
                  names[i], ic, INTG((*If[j]), ivals[2*j], ivals[2*j+1]));
        }
        clock_t end = clock();
        timeSpentParallel= (double)(end - begin) / CLOCKS_PER_SEC;
        printf("\n");
        printf("time it takes parallel: %f\n", timeSpentParallel);
      }

      clock_t begin1 = clock();
      for(j=0; j < 4; j++)
      {
        for(i=1; i < 2; i++)
        {
          ic = (*f[i])(ivals[2*j], ivals[2*j+1], approx[j], rf[j]);
          printf("%10s [ 0,1] num: %+lf, an: %lf\n",
                  names[i], ic, INTG((*If[j]), ivals[2*j], ivals[2*j+1]));
        }
        clock_t end1= clock();
        timeSpentSequential = (double)(end1 - begin1) / CLOCKS_PER_SEC;
        printf("\n");
        printf("time it takes normal: %f\n", timeSpentSequential);
       
      }
       checkDifference(timeSpentParallel, timeSpentSequential);
}
void executeLeftRect(rfunc rf[],rfunc If[],double ivals[],double approx[]){
  pfunc f[2] = {intlLeftRectParallel , int_leftrect};
  const char *names[2] = {"LeftRectParallel" , "LeftRectSequential"};
  iteratesThroughTwoFunctions(rf, If, ivals, approx, f, names);
}
void executeRightRect(rfunc rf[],rfunc If[],double ivals[],double approx[]){
  pfunc f[2] = {intRightrectParallel, int_rightrect};
  const char *names[2] = {"RightRectParallel" , "RightRectSequential" };
 iteratesThroughTwoFunctions(rf, If, ivals, approx, f, names);
}

void executeMidRect(rfunc rf[],rfunc If[],double ivals[],double approx[]){
  pfunc f[2] = {intMidrectParallel, int_midrect};
  const char *names[2] = {"MidRectParallel" , "MidRectSequential"};
  iteratesThroughTwoFunctions(rf, If, ivals, approx, f, names);
}

void executeTrapezium(rfunc rf[],rfunc If[],double ivals[],double approx[]){
  pfunc f[2] = {intTrapeziumParallel, int_trapezium};
  const char *names[2] = {"TrapeziumParallel" , "TrapeziumSequential"};
  iteratesThroughTwoFunctions(rf, If, ivals, approx, f, names);
}
void executeSimpson( rfunc rf[],rfunc If[],double ivals[],double approx[]){
  pfunc f[2] = {intSimpsonParallel , int_simpson};
  const char *names[2] = {"SimpsonParallel" , "SimpsonSequential"};
  iteratesThroughTwoFunctions(rf, If, ivals, approx, f, names);
}



char menu(){
  char decision;
  printf("Which method would you like to use to integrate: \n");
  printf("1).  leftRect \n");
  printf("2).  rightRect \n");
  printf("3).  midRect \n");
  printf("4).  Trapezium \n");
  printf("5).  Simpson \n");
  printf("6). Stop \n");
  scanf(" %c", &decision);

  if(decision !=  1){
      return decision;
  }else 
    return EOF;
}

int main()
{
    clock_t t; 
     rfunc rf[] = { f1, f2, f3, f3 };
     rfunc If[] = { f1a, f2a, f3a, f3a };
     double ivals[] = { 
      0.0, 1.0,
       1.0, 100.0,
       0.0, 5000.0,
       1.0, 600000.0
     };
     double approx[] = { 1000000.0, 100000.0, 5000000.0 , 60000000.0 };
     char decision;
     do{
       decision = menu();
      switch (decision)
      {
      case '1':
          executeLeftRect(rf,If,ivals,approx);
        break;
      case '2':
       
          executeRightRect(rf,If,ivals,approx);
        break;
      case '3':
          executeMidRect(rf,If,ivals,approx);
        break;
      case '4': 
          executeTrapezium(rf,If,ivals,approx);
        break;  
      case '5':
          executeSimpson(rf,If,ivals,approx);
        break;
      case '6' :
          printf("program has being terminated correctly");
      case EOF:
        break;
      default: 
        printf("invalid option");
        break;
      }
     

     }while(decision != '6'); 

 

}