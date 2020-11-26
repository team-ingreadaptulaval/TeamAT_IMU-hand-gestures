cmake_minimum_required(VERSION 3.15)
project(windows_cpp_client_socket)

set(CMAKE_CXX_STANDARD 14)

add_executable(windows_cpp_client_socket main.cpp utils.cpp utils.h Socket.cpp Socket.h)