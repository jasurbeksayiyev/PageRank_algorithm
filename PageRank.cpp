#include <vector>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Function to calculate the PageRank
void pageRank(vector<double>& ranks, const vector<vector<int>>& links, int iterations, double d = 0.85, double tol = 1e-6) {
    int n = ranks.size();
    vector<double> new_ranks(n, 1.0 / n);
    vector<double> prev_ranks = ranks;
    bool converged = false;

    for (int it = 0; it < iterations && !converged; it++) {
        fill(new_ranks.begin(), new_ranks.end(), (1.0 - d) / n);

#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            for (int j : links[i]) {
#pragma omp atomic
                new_ranks[j] += d * ranks[i] / links[i].size();
            }
        }

        // Check for convergence
        converged = true;
#pragma omp parallel for reduction(&:converged)
        for (int i = 0; i < n; i++) {
            if (fabs(new_ranks[i] - prev_ranks[i]) > tol) {
                converged = false;
            }
        }

        ranks.swap(new_ranks);
        prev_ranks = ranks;

        // Print intermediate ranks
        cout << "Iteration " << it + 1 << ": ";
        for (double rank : ranks) {
            cout << rank << " ";
        }
        cout << endl;
    }
}

int main() {
    vector<vector<int>> links = {{1, 2}, {2}, {0}};  // Example link structure
    vector<double> ranks(3, 1.0 / 3);  // Initial ranks

    // Measure execution time
    auto start = high_resolution_clock::now();

    pageRank(ranks, links, 100);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    // Print final ranks
    cout << "\nFinal Ranks:\n";
    for (double rank : ranks) {
        cout << rank << endl;
    }

    cout << "\nExecution Time: " << duration.count() << " milliseconds" << endl;

    return 0;
}
