#include <iostream>

int main()
{
#if defined(__ARM_NEON) && defined(__aarch64__)
    std::cout << "Finally succ! Congratulation!" << std::endl;
#endif
    std::cout << "Finally succ! Congratulation!" << std::endl;
}
