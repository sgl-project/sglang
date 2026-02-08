#include <iostream>
#include <string>

extern std::string foo();

int main() { std::cout << foo() << std::endl; }
