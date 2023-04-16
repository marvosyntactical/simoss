#include <cmath>
#include <vector>
#include <algorithm>

using namespace std;

// Function to compute the eigenvalue decomposition of a matrix
void eigen(float** A, int n, float**& eig_vecs, float*& eig_vals) {
  // Initialize the matrix of eigenvectors
  eig_vecs = new float*[n];
  for (int i = 0; i < n; i++) {
      eig_vecs[i] = new float[n];
      for (int j = 0; j < n; j++) {
          // kronecker delta
          eig_vecs[i][j] = (i == j) ? 1.0f : 0.0f;
      }
  }

  // Initialize the diagonal matrix of eigenvalues
  eig_vals = new float[n];
  for (int i = 0; i < n; i++) {
    eig_vals[i] = A[i][i];
  }

  // Set a tolerance level for convergence
  float tol = 1e-2;

  // Iterate until convergence
  while (true) {
    // Apply a QR decomposition to the matrix of eigenvectors
    for (int k = 0; k < n - 1; k++) {
      // Compute the Householder reflector
      float alpha = 0.0f;
      for (int i = k; i < n; i++) {
        alpha += eig_vecs[i][k] * eig_vecs[i][k];
      }
      alpha = sqrt(alpha);
      if (eig_vecs[k][k] < 0) {
        alpha = -alpha;
      }
      float beta = alpha * (alpha - eig_vecs[k][k]);
      eig_vecs[k][k] -= alpha;

      // Apply the reflector to the remaining columns
      for (int j = k + 1; j < n; j++) {
        float gamma = 0.0f;
        for (int i = k; i < n; i++) {
          gamma += eig_vecs[i][k] * eig_vecs[i][j];
        }
        gamma /= beta;
        for (int i = k; i < n; i++) {
          eig_vecs[i][j] -= gamma * eig_vecs[i][k];
        }
      }

      // Apply the reflector to the matrix of eigenvalues
      float delta = 0.0f;
      for (int i = k; i < n; i++) {
        delta += A[i][k] * A[i][k];
      }
      delta = sqrt(delta);
      if (A[k][k] < 0) {
        delta = -delta;
      }
      float epsilon = delta * (delta - A[k][k]);
      A[k][k] -= delta;
      for (int j = k + 1; j < n; j++) {
        float zeta = 0.0f;
        for (int i = k; i < n; i++) {
          zeta += A[i][k] * A[i][j];
        }
        zeta /= epsilon;
        for (int i = k; i < n; i++) {
          A[i][j] -= zeta * A[i][k];
        }
      }
    }

    // Check for convergence
    float delta = 0.0f;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            delta += fabs(A[i][j]);
        }
    }
    if (delta < tol) {
	    break;
	    }
    }

    // Sort the eigenvalues in decreasing order and rearrange the eigenvectors accordingly
    vector<pair<float, int>> pairs(n);
    for (int i = 0; i < n; i++) {
        pairs[i] = make_pair(eig_vals[i], i);
    }
    sort(pairs.begin(), pairs.end(), greater<pair<float, int>>());
    for (int i = 0; i < n; i++) {
        eig_vals[i] = pairs[i].first;
        int idx = pairs[i].second;
        for (int j = 0; j < n; j++) {
            swap(eig_vecs[j][idx], eig_vecs[j][i]);
        }
    }
}

// Function to compute the square root of a positive semidefinite matrix
void sqrtm(float** C, int n, float**& C_sqrt) {
    // Compute the eigenvalue decomposition of the matrix
    float** eig_vecs;
    float* eig_vals;
    eigen(C, n, eig_vecs, eig_vals);

    // Compute the square root of the eigenvalues
    float* eig_vals_sqrt = new float[n];
    for (int i = 0; i < n; i++) {
        eig_vals_sqrt[i] = sqrt(max(0.0f, eig_vals[i]));
    }

    // Compute the square root of the matrix using the formula C^(1/2) = V * sqrt(D) * V^-1
    C_sqrt = new float*[n];
    for (int i = 0; i < n; i++) {
        C_sqrt[i] = new float[n];
        for (int j = 0; j < n; j++) {
            C_sqrt[i][j] = 0.0f;
            for (int k = 0; k < n; k++) {
                float v = eig_vecs[i][k] * eig_vals_sqrt[k] * eig_vecs[j][k];
                C_sqrt[i][j] += v;
                cout << "val = " << v << "\n";
            }
        }
    }

    // Free memory
    for (int i = 0; i < n; i++) {
        delete[] eig_vecs[i];
    }
    delete[] eig_vecs;
    delete[] eig_vals;
    delete[] eig_vals_sqrt;
}
