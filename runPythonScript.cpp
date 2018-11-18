#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

std::string exec(const char* cmd) {
    std::array<char, 100000> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
            result += buffer.data();
    }
    return result;
}

int main(int argc, char** argv)
{
    char buffer[200];
    sprintf(buffer, "python3 %s %s %s %s %s %s \n", argv[argc - 6],argv[argc - 5], 
                                                    argv[argc - 4], argv[argc - 3], 
                                                    argv[argc - 2], argv[argc - 1]);
    std::string output = exec(buffer);
    return 0;
}