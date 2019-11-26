#include "solverObject.h"
int main() {
    int max  = omp_get_max_threads();
    for (int i = 1; i <= max; ++i) {
        omp_set_num_threads(i);
        unsigned int N{400};/*size of mesh*/
        solverObject *obj = new solverObject(N, i);
        //std::unique_ptr<MatrixObj> obj(new MatrixObj{512,16, i});
        obj -> runRedAndBlackIterations();
        obj -> runJacoby();
        delete obj;
    }
    return 0;
}