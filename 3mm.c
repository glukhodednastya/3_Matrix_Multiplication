// /* Include benchmark-specific header. */
#include <omp.h>
#include "3mm.h"
#include "general_funcs.h"

static
void kernel_3mm(int ni, int nj, int nk, int nl, int nm,
                double E[ ni][nj], double A[ ni][nk],
                double B[ nk][nj], double F[ nj][nl],
                double C[ nj][nm], double D[ nm][nl],
                double G[ ni][nl])
{
  int i, j, k;
  
  // #pragma omp parallel shared(E, A, B)
  #pragma omp parallel default(shared) private(i, j, k)
  {
    #pragma omp for
    for (i = 0; i < ni; i++) {
      for (j = 0; j < nj; j++) {
        E[i][j] = 0.0;
        for (k = 0; k < nk; ++k) {
          E[i][j] += A[i][k] * B[k][j];
        }
      }
    }
  }

  #pragma omp parallel shared(F, C, D) private(i, j, k)
  {
   #pragma omp for
    for (i = 0; i < nj; i++) {
      for (j = 0; j < nl; j++) {
        F[i][j] = 0.0;
        for (k = 0; k < nm; ++k) {
          F[i][j] += C[i][k] * D[k][j];
        }
      }
    }
  }

  #pragma omp parallel shared(G, E, F) private(i, j, k)
  {
    #pragma omp for
    for (i = 0; i < ni; i++) {
      for (j = 0; j < nl; j++) {
        G[i][j] = 0.0;
        for (k = 0; k < nj; ++k) {
          G[i][j] += E[i][k] * F[k][j];
        }
      }
    }
  }

}

int CheckEq(int ni, int nj, double LHS[ni][nj], double RHS[ni][nj]) {
  for (int i = 0; i < ni; ++i) {
    for (int k = 0; k < nj; ++k) {
      if (LHS[i][k] != RHS[i][k]) {
        return 0;
      }
    }
  }
  return 1;
}

int main(int argc, char** argv) {
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;
  int nm = NM;

  double (*E)[ni][nj]; E = (double(*)[ni][nj])malloc ((ni) * (nj) * sizeof(double));
  double (*A)[ni][nk]; A = (double(*)[ni][nk])malloc ((ni) * (nk) * sizeof(double));
  double (*B)[nk][nj]; B = (double(*)[nk][nj])malloc ((nk) * (nj) * sizeof(double));
  double (*F)[nj][nl]; F = (double(*)[nj][nl])malloc ((nj) * (nl) * sizeof(double));
  double (*C)[nj][nm]; C = (double(*)[nj][nm])malloc ((nj) * (nm) * sizeof(double));
  double (*D)[nm][nl]; D = (double(*)[nm][nl])malloc ((nm) * (nl) * sizeof(double));
  double (*G)[ni][nl]; G = (double(*)[ni][nl])malloc ((ni) * (nl) * sizeof(double));

  init_array (ni, nj, nk, nl, nm, *A, *B, *C, *D);
  bench_timer_start();
  kernel_3mm (ni, nj, nk, nl, nm, *E, *A, *B, *F, *C, *D, *G);
  bench_timer_stop();
  bench_timer_print();

  if (argc > 42 && ! strcmp(argv[0], "")) print_array(ni, nl, *G);

  free((void*)E);
  free((void*)A);
  free((void*)B);
  free((void*)F);
  free((void*)C);
  free((void*)D);
  free((void*)G);

  return 0;
}
