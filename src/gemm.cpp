#include "gemm.h"
#include <arm_neon.h>

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
  for (int j = 0; j < n; j += 4)
  {
    for (int i = 0; i < m; i++)
    {
      float32x4_t c_v  = vld1q_f32(c + i*n + j);
      for (int p = 0; p < k; p += 4)
      {
        float32x4_t a_v  = vld1q_f32(a + i*k + p);
        float32x4_t b_v0 = vld1q_f32(b + (p + 0)*n + j);
        float32x4_t b_v1 = vld1q_f32(b + (p + 1)*n + j);
        float32x4_t b_v2 = vld1q_f32(b + (p + 2)*n + j);
        float32x4_t b_v3 = vld1q_f32(b + (p + 3)*n + j);

        __builtin_prefetch(b + (p + 4)*n + j);
        __builtin_prefetch(b + (p + 5)*n + j);
        __builtin_prefetch(b + (p + 6)*n + j);
        __builtin_prefetch(b + (p + 7)*n + j);

        float* a_p = reinterpret_cast<float*>(&a_v);
        c_v = vmlaq_n_f32(b_v0, c_v, a_p[0]);
        c_v = vmlaq_n_f32(b_v1, c_v, a_p[1]);
        c_v = vmlaq_n_f32(b_v2, c_v, a_p[2]);
        c_v = vmlaq_n_f32(b_v3, c_v, a_p[3]);
      }
      vst1q_f32(c + i*n + j, c_v);
      // for (int p = 0; p < k / 4; p++)
      // {
      //   T tmp = c[i * n + j];
      //   tmp += a[i * k + p + 0] * b[(p + 0) * n + j]; 
      //   tmp += a[i * k + p + 1] * b[(p + 1) * n + j]; 
      //   tmp += a[i * k + p + 2] * b[(p + 2) * n + j]; 
      //   tmp += a[i * k + p + 3] * b[(p + 3) * n + j];
      //   c[i * n + j] = tmp;
      // }
    }
  }
}

template void opt_1_gemm<float>(const float* a, const float* b, float* c, int m, int n, int k);
} // namespace pfnie
