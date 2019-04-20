cmake_minimum_required(VERSION 3.6)

project(VulkanRenderer VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include("compiler")
include("clangformat")

if (BP_ENABLE_CLANG_TIDY)
include("ClangTidy")
endif()

if (BP_ENABLE_CPPCHECK)
include("cppcheck")
endif()

if(MSVC)
    add_compile_definitions(NOMINMAX)
endif()

add_subdirectory(third-party)
add_subdirectory(src)
