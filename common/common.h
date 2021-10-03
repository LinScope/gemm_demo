#include <iostream>
#include <vector>

#include "timer.h"
#include "device.h"

using namespace std;

namespace pfnie
{

template <typename T>
bool compare_array(const T* ary_a, const T* ary_b, size_t length)
{
  // TODO: impl more features（for debug and compare. e.g. diff rate and diff range)
  //return std::memcmp((void*)ary_a, (void*)ary_b, length * sizeof(T));
  for (int i = 0; i < length; i++)
  {
    if (ary_a[i] != ary_b[i]) 
    {
      std::cout << "a, b = " << ary_a[i] << ", " << ary_b[i] << std::endl;
      return false;
    }
  }

  return true;
}

} // namespace pfnie
