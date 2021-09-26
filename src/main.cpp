#include <cstring>

#include "../common/common.h"
#include "gemm.h"

using namespace pfnie;

void device_memory_init(void* data, size_t length, Device dev)
{
  if (dev.type == DeviceType::Cpu)
    memset(data, 1, length);
  if (dev.type == DeviceType::Cuda)
    printf("Error, not impl\n");
}

int main(int argv, char* argc[])
{
  using dtype = float;

  const int iter  = 100;
  const int count = 10;

  const int M = 200;
  const int K = 200;
  const int N = 200;

  bool define_compare = true;
  //if (argv > 1）define_compare = atoi(argc[1]);

  vector<dtype*> array_a(count, nullptr);
  vector<dtype*> array_b(count, nullptr);
  vector<dtype*> array_c(count, nullptr);
  vector<dtype*> array_o(count, nullptr);

  // malloc
  for (int i = 0; i < count; i++)
  {
    // cpu only
    array_a[i] = (dtype*)malloc(M * K * sizeof(dtype));
    array_b[i] = (dtype*)malloc(K * N * sizeof(dtype));
    array_c[i] = (dtype*)malloc(M * N * sizeof(dtype));
    array_o[i] = (dtype*)malloc(M * N * sizeof(dtype));
  }

  // init
  for (int i = 0; i < count; i++)
  {
    device_memory_init((void*)array_a[i], M * K * sizeof(dtype), {DeviceType::Cpu, -1});
    device_memory_init((void*)array_b[i], K * N * sizeof(dtype), {DeviceType::Cpu, -1});
  }
  
  for (int i = 0; i < count; i++)
  {
    generic_gemm<dtype>(array_a[i % count], 
                        array_b[i % count], 
                        array_c[i % count], 
                        M, K, N);
  }

  chrono::milliseconds std_duration = milliseconds::zero();
  chrono::milliseconds opt_duration = milliseconds::zero();
  for (int i = 0; i < iter; i++)
  {
    auto&& ck0 = chrono::high_resolution_clock::now();
    
    generic_gemm<dtype>(array_a[i % count], 
                        array_b[i % count], 
                        array_c[i % count], 
                        M, K, N);
        
    auto&& ck1 = chrono::high_resolution_clock::now();
    std_duration += duration_cast<milliseconds>(ck1 - ck0);

    auto&& ck2 = chrono::high_resolution_clock::now();
    
    generic_gemm<dtype>(array_a[i % count], 
                        array_b[i % count], 
                        array_o[i % count], 
                        M, K, N);
    
    auto&& ck3 = chrono::high_resolution_clock::now();
    opt_duration += duration_cast<milliseconds>(ck3 - ck2);

    if (define_compare)
    {
      if (compare_array<dtype>(array_c[i % count], array_o[i % count], M * N) == false)
      {
        std::cout << "gemm result error, not match!" << std::endl;
      }
    }
  }

  auto&& std_ms = float(std_duration.count()) / iter;
  auto&& opt_ms = float(opt_duration.count()) / iter;

  printf("std gemm time cost = %f ms (%d x %d x %d - %d iter)\n",
      std_ms, M, K, N, iter);
  printf("opt gemm time cost = %f ms (%d x %d x %d - %d iter)\n",
      opt_ms, M, K, N, iter);
  
  // free 
  for (int i = 0; i < count; i++)
  {
    // cpu only
    free(array_a[i]);
    free(array_b[i]);
    free(array_c[i]);
    free(array_o[i]);
  }
}
