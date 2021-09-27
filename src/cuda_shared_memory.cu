
__global__ void static_shared_memory()
{
  __shared__ float array[1024];
}

__global__ void dynamic_shared_memory()
{
  __shared__ float array[];
}

int launcher()
{
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // static
  static_shared_memory<<<256, 256, 0, stream>>>();
  
  // dynamic
  dynamic_shared_memory<<<256, 256, 1024 * sizeof(int), stream>>>();

  cudaStreamSynchronize(stream);
  cudaStreamDestory(&stream);
}
