#include <iostream>

int fn(int a, int b, double c)
{
  int d = a + b;
  return d - c;
}

int main()
{
  const int a = 1996;
  int b = 1971;
  double c = 1949;
  std::cout << fn(a, b, c) << std::endl;
  return 0;
}
