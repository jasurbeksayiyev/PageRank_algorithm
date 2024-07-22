#include <omp.h> // Include OpenMP library

// Parallel region inside the iteration loop
for (int iter = 0; iter < iterations; iter++) {
vector<double> newPR(N, (1 - d) / N);
#pragma omp parallel for
for (int i = 0; i < N; i++) {
double sum = 0.0;
for (int j : links[i]) {
#pragma omp atomic
newPR[j] += d * PR[i] / links[i].size();
}
}
if (norm(newPR, PR) < 1e-6) break; // Convergence check
PR = newPR;
printPageRank(PR);
}
