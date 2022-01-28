cmake_minimum_required(VERSION 3.20)
project(SmartFileSystem)

set(CMAKE_CXX_STANDARD 14)

add_executable(SmartFileSystem main.cpp SmartBuffer.cpp SmartBuffer.h SmartFS.cpp SmartFS.h Error.cpp Error.h Tool.cpp Tool.h SmartDataStructure.cpp SmartDataStructure.h SmartDiskDriver.cpp SmartDiskDriver.h)
