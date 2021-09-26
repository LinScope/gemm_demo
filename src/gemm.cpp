#include "gemm.h"

namespace pfnie
{

template <typename T>
void generic_gemm(
    const T* a,
    const T* b,
    T* c,
    const int m,
    const int k,
    const int n)
{
  // TODO: without memory init
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      for (int p = 0; p < k; p++)
      {
        c[i * n + j] += a[i * k + p] * b[p * n + j]; 
      }
    }
  }
}

template void generic_gemm<float>(const float* a, const float* b, float* c, int m, int n, int k);

template <typename T>
void opt_1_gemm(
     const T* a,
     const T* b,
     T* c,
     const int m,
     const int k,
     const int n)
{
 for (int i = 0; i < m; i++)
 {
   for (int j = 0; j < n; j++)
   {
     for (int p = 0; p < k / 4; p++)
     {
       c[i * n + j] += a[i * k + p + 0] * b[(p + 0) * n + j]; 
       c[i * n + j] += a[i * k + p + 1] * b[(p + 1) * n + j]; 
       c[i * n + j] += a[i * k + p + 2] * b[(p + 2) * n + j]; 
       c[i * n + j] += a[i * k + p + 3] * b[(p + 3) * n + j]; 
     }
   }
 }
}

template void opt_1_gemm<float>(const float* a, const float* b, float* c, int m, int n, int k);
} // namespace pfnie
