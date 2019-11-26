#include "solverObject.h"

void solverObject::setMatrixFromFile(std::string fileName){
    auto fileeName = "/Users/kola/Desktop/LAB/inputFile.txt";
std::ifstream inputFile(fileName,std::ios_base::in);
double temp{0};
for(std::size_t i = 0; i < N; i++)
{
inputFile >> temp;
//Matrix.push_back(temp);
}
inputFile.close();
}

double solverObject::calculateAnalyticalError(const std::vector<double> &Solution) {
    double temp{0}, max{0};
#pragma omp parallel for default(none) shared(Solution, max) private(temp)
    for (unsigned int i = 0; i < N; ++i) {
        for (unsigned int j = 0; j < N; ++j) {
            temp = fabs((Solution[i*N + j]) - analyticalSolution(i*h, j*h));
            if(temp > max){
#pragma omp critical
                if (temp > max)
                    max = temp;
                }
            }
        }
    return max;
}

double solverObject::calculateLocalError(const std::vector<double> &oldSolution, const std::vector<double> &newSolution) {
    double temp{0}, max{0};
#pragma omp parallel for default(none) shared(oldSolution, newSolution, max) private(temp)
    for (unsigned int i = 0; i < N; ++i) {
        for (unsigned int j = 0; j < N; ++j) {
            temp = fabs((newSolution[i*N + j]) - oldSolution[i*N + j]);
            if(temp > max){
#pragma omp critical
                if (temp > max)
                    max = temp;
            }
        }
    }
    return max;
}

void solverObject::runRedAndBlackIterations() {
    std::cout << "Red-black iterations method." << std::endl;
    std::vector<double> newSolution;
    std::vector<double> oldSolution;
    newSolution.reserve(N*N);
    oldSolution.reserve(N*N);
    double denominator = 1.0/(4 + lambda*lambda*h*h);
    double error{0.0};
    unsigned int iterationsCounter{0};
    /*Initializes vectors with boundary conditions */
#pragma omp parallel for default(none) shared(oldSolution, newSolution)
    for (unsigned int i = 0; i < N*N; ++i) {
        oldSolution.push_back(0.0);
        newSolution.push_back(0.0);
    }
    double t1 = omp_get_wtime();
    while(true) {
        iterationsCounter++;
        /*do red iterations*/
#pragma omp parallel for default(none) shared(oldSolution, newSolution, denominator)
        for (unsigned int i = 1; i < N - 1; ++i) {
            for (unsigned int j = (i % 2 + 1); j < N - 1; j += 2) {
                newSolution[i*N + j] = denominator * (h * h * f(i * h, j * h) + oldSolution[(i + 1) * N + j] +
                                                   oldSolution[(i - 1) * N + j] + oldSolution[i * N + j + 1] + oldSolution[i * N + j - 1]);
            }
        }
        /*do black iterations*/
#pragma omp parallel for default(none) shared(newSolution, denominator)
        for (unsigned int i = 1; i < N - 1; ++i) {
            for (unsigned int j = ((i + 1) % 2 + 1); j < N - 1; j += 2) {
                newSolution[i*N + j] = denominator * (h * h * f(i * h, j * h) + newSolution[(i + 1) * N + j] +
                                                   newSolution[(i - 1) * N + j] + newSolution[i * N + j + 1] + newSolution[i * N + j - 1]);
            }
        }
        error = calculateLocalError(newSolution, oldSolution);
        if (error < eps){
            std::cout << "Analytical error: " << calculateAnalyticalError(newSolution) << std::endl;
            break;
        }
        /*If local error is smaller than parameter*/
        oldSolution.swap(newSolution);
    }
    double t2 = omp_get_wtime();
    std::cout << "Parallel algorithm took : " << (t2 - t1) << " seconds"<< std::endl;
    std::cout << "Number of iterations: " << iterationsCounter << std::endl;
    std::cout << "Error: " << error << std::endl;
}

void solverObject::runJacoby() {
    std::cout << "Jacoby method." << std::endl;
    std::vector<double> newSolution;
    std::vector<double> oldSolution;
    newSolution.reserve(N*N);
    oldSolution.reserve(N*N);
    double denominator = 1.0/(4 + lambda*lambda*h*h);
    double error{0.0};
    unsigned int iterationsCounter{0};
    /*Initializes vectors with boundary conditions */
#pragma omp parallel for default(none) shared(oldSolution, newSolution)
    for (unsigned int i = 0; i < N*N; ++i) {
        oldSolution.push_back(0.0);
        newSolution.push_back(0.0);
    }

    double t1 = omp_get_wtime();
    while(true) {
        iterationsCounter++;
#pragma omp parallel for default(none) shared(oldSolution, newSolution, denominator)
        for (unsigned int i = 1; i < N - 1; ++i) {
            for (unsigned int j = 1; j < N - 1; j++) {
                newSolution[i*N + j] = denominator * (h * h * f(i * h, j * h) + oldSolution[(i + 1) * N + j] +
                                                      oldSolution[(i - 1) * N + j] + oldSolution[i * N + j + 1] + oldSolution[i * N + j - 1]);
            }
        }
        error = calculateLocalError(newSolution, oldSolution);
        if (error < eps){
            std::cout << "Analytical error: " << calculateAnalyticalError(newSolution) << std::endl;
            break;
        }
        /*If local error is smaller than parameter*/
        oldSolution.swap(newSolution);
    }
    double t2 = omp_get_wtime();
    std::cout << "Parallel algorithm took : " << (t2 - t1) << " seconds"<< std::endl;
    std::cout << "Number of iterations: " << iterationsCounter << std::endl;
    std::cout << "Error: " << error << std::endl;
}

