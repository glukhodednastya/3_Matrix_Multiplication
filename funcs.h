#pragma once

/* Include benchmark-specific header. */
#include <omp.h>

#include "3mm.h"

double bench_t_start, bench_t_end;

// setenv OMP_NESTED true

static
double rtclock() {
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0) {
      printf ("Error return from gettimeofday: %d", stat);
    }
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start() {
  bench_t_start = rtclock ();
}

void bench_timer_stop() {
  bench_t_end = rtclock ();
}

void bench_timer_print() {
  printf ("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}

// static
// void init_(int ni, int nk, double A[ ni][nk], int start) {
//   for (int i = start; i < ni + start; i++) {
//     for (int j = start * 2; j < nk + start * 2; j++) {
//       A[i - start][j - start * 2] = (double) ((i*j+1) % ni) / (nk * 5) + rand() % 10;
//     }
//   }
// }

static
void init_array(int ni, int nj, int nk, int nl, int nm,
                double A[ ni][nk], double B[ nk][nj],
                double C[ nj][nm], double D[ nm][nl])
{
  int i, j;

  #pragma omp parallel shared(A) private(i, j)
  {
    #pragma omp for
    for (i = 0; i < ni; i++) {
      for (j = 0; j < nk; j++) {
        A[i][j] = (double) ((i*j+1) % ni) / (5*ni);
      }
    }
  }

  #pragma omp parallel shared(B) private(i, j)
  {
    #pragma omp for
    for (i = 0; i < nk; i++) {
      for (j = 0; j < nj; j++) {
        B[i][j] = (double) ((i*(j+1)+2) % nj) / (5*nj);
      }
    }
  }

  #pragma omp parallel shared(C) private(i, j)
  {
    #pragma omp for
    for (i = 0; i < nj; i++) {
      for (j = 0; j < nm; j++) {
        C[i][j] = (double) (i*(j+3) % nl) / (5*nl);
      }
    }
  }

  #pragma omp parallel shared(D) private(i, j)
  {
    #pragma omp for
    for (i = 0; i < nm; i++) {
      for (j = 0; j < nl; j++) {
        D[i][j] = (double) ((i*(j+2)+2) % nk) / (5*nk);
      }
    }
  }
}

static
void print_array(int ni, int nl, double G[ ni][nl]) {
  int i, j;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "G");
  for (i = 0; i < ni; i++) {
    for (j = 0; j < nl; j++) {
      if ((i * ni + j) % 20 == 0) {
        fprintf (stderr, "\n");
      }
      fprintf (stderr, "%0.2lf ", G[i][j]);
    }
  }
  fprintf(stderr, "\nend   dump: %s\n", "G");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

void sum(int nl, int nr, double A[ nl][nr], double B[ nl][nr], double Res[nl][nr]) {
  for (int i = 0; i < nl; ++i) {
    for (int k = 0; k < nr; ++k) {
      Res[i][k] = A[i][k] + B[i][k];
    }
  }
}