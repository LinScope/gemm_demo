
namespace pfnie
{

template <typename T>
void generic_gemm(
    const T* a,
    const T* b,
    T* c,
    const int m,
    const int k,
    const int n);

template <typename T>
void opt_1_gemm(
    const T* a,
    const T* b,
    T* c,
    const int m,
    const int k,
    const int n);

} // namespace pfnie
