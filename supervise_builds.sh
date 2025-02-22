find . -name "*.cpp" -o -name "*.h" -maxdepth 2 | entr -s "cmake --build build"
