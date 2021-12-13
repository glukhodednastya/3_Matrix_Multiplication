#include <omp.h>
#include "3mm.h"
#include "funcs.h"

void add(int nl, int nr, double A[ nl][nr], double Res[nl][nr]) {
  for (int i = 0; i < nl; ++i) {
    for (int k = 0; k < nr; ++k) {
      Res[i][k] += A[i][k];
    }
  }
}

void usual_mult(int nl, int nc, int nr, double A[ nl][nc], double B[ nc][nr], double Res[nl][nr]) {
  int i, j, k;
  #pragma omp parallel default(shared) private(i, j, k)
  {
    #pragma omp for
    for (i = 0; i < nl; i++) {
      for (j = 0; j < nr; j++) {
        Res[i][j] = 0;
        for (k = 0; k < nc; ++k) {
          Res[i][j] += A[i][k] * B[k][j];
        }
      }
    }
  }
}


void block_mult(int nl, int nc, int nr, double A[ nl][nc], double B[ nc][nr], double Res[nl][nr]) {
  int A_ln_size = 200;
  int c_size = 200;
  int B_cols_size = 200;

  int lines_am = nl / A_ln_size + (nl % A_ln_size != 0);
  int cols_am = nr / B_cols_size + (nr % B_cols_size != 0);
  int res_c_size = nc / c_size + (nc % c_size != 0);

  int i, k;

  #pragma omp parallel default(shared) private(i, k)
  {
    #pragma omp for
    for (i = 0; i < lines_am; ++i) {
      for (k = 0; k < cols_am; ++k) {
        int ln = nl - A_ln_size * i > A_ln_size ? A_ln_size : nl - A_ln_size * i;
        int cl = nr - B_cols_size * k > B_cols_size ? B_cols_size : nr - B_cols_size * k;

        double (*temp)[ln][cl]; temp = (double(*)[ln][cl])malloc ((ln) * (cl) * sizeof(double));
		
        for (int s = 0; s < ln; ++s) {
          for (int j = 0; j < cl; ++j) {
            (*temp)[s][j] = 0;
          }
        }
        for (int s = 0; s < res_c_size; ++s) {
			
          int cs = nc - c_size * s > c_size ? c_size : nc - c_size * s;
          double (*temp_A)[ln][cs]; temp_A = (double(*)[ln][cs])malloc ((ln) * (cs) * sizeof(double));
          double (*temp_B)[cs][cl]; temp_B = (double(*)[cs][cl])malloc ((cs) * (cl) * sizeof(double));
		  
          for (int r = 0; r < ln; ++r) {
            for (int g = 0; g < cs; ++g) {
              (*temp_A)[r][g] = A[r + A_ln_size * i][g + c_size * s];
            }
          }
          for (int g = 0; g < cs; ++g) {
            for (int r = 0; r < cl; ++r) {
              (*temp_B)[g][r] = B[g + c_size * s][r + B_cols_size * k];
            }
          }
          double (*part_temp)[ln][cl]; part_temp = (double(*)[ln][cl])malloc ((ln) * (cl) * sizeof(double));
          usual_mult(ln, cs, cl, *temp_A, *temp_B, *part_temp);
          add(ln, cl, *part_temp, *temp);
        }

        for (int r = 0; r < ln; ++r) {
          for (int g = 0; g < cl; ++g) {
            Res[r + A_ln_size * i][g + B_cols_size * k] = (*temp)[r][g];
          }
        }
      }
    }
  }
}

static
void kernel_3mm_bl(int ni, int nj, int nk, int nl, int nm,
                   double E[ ni][nj], double A[ ni][nk],
                   double B[ nk][nj], double F[ nj][nl],
                   double C[ nj][nm], double D[ nm][nl],
                   double G[ ni][nl]) {
  
  block_mult(ni, nk, nj, A, B, E);
  block_mult(nj, nm, nl, C, D, F);
  block_mult(ni, nj, nl, E, F, G);
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

  kernel_3mm_bl (ni, nj, nk, nl, nm, *E, *A, *B, *F, *C, *D, *G);

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
