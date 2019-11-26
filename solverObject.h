#ifndef GELMGOLZEQUATIONSOLVER_SOLVEROBJECT_H
#define GELMGOLZEQUATIONSOLVER_SOLVEROBJECT_H
#pragma once
#include <fstream>
#include <iostream>
#include <iomanip>
#include "omp.h"
#include <cstdlib>
#include <vector>

class solverObject {

public:
   solverObject(unsigned int N, int i): N(N), h(1.0/(N-1)), lambda(static_cast<double>(N)){
        std::cout << "Amount of threads: " << i << std::endl;
    };
    ~solverObject(){
        std::cout << std::endl;
    };

    void setMatrixFromFile(std::string fileName);
    inline double f(const double &x, const double &y); /*returns value of right side in point*/
    inline double analyticalSolution(const double &x, const double &y); /*returns analytical solution in point*/
    double calculateAnalyticalError(const std::vector<double> &Solution);/*A[i][j] - AnalyticalSoluyion(x_i, y_i)*/
    double calculateLocalError(const std::vector<double> &oldSolution,const std::vector<double> &newSolution );/*oldSolution[i,j] - newSolution[i,j]*/
    void runRedAndBlackIterations();/*runs red-black iterations*/
    void runJacoby();/*runs Jacoby iterations*/
private:
    unsigned int N;/*size of mesh*/
    double h;/*mesh step*/
    double lambda;
    double eps{10e-6};
};


double solverObject::f(const double &x, const double &y){
    return 2.0 * sin(M_PI * y) + lambda * lambda * (1.0 - x) * x * sin(M_PI * y) \
    + M_PI * M_PI * (1.0 - x) * x * sin(M_PI * y);
}
double solverObject::analyticalSolution(const double &x, const double &y){
    return (1.0 - x) * x * sin(M_PI * y);;
}


#endif //GELMGOLZEQUATIONSOLVER_SOLVEROBJECT_H
