find_package(Vulkan)

add_executable(VulkanRenderer "main.cpp"
    "platform.hpp" "platform.cpp"
    "shader_module.hpp" "shader_module.cpp")
target_link_libraries(VulkanRenderer
    PRIVATE compiler_warnings
    Vulkan::Vulkan
    CONAN_PKG::glfw
    )

set_target_properties(VulkanRenderer PROPERTIES RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/bin")
add_dependencies(VulkanRenderer assets)
