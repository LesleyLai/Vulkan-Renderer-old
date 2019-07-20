find_package(Vulkan)

add_executable(VulkanRenderer "main.cpp"
    "platform.hpp" "platform.cpp"
    "shader_module.hpp" "shader_module.cpp")
target_link_libraries(VulkanRenderer
    PRIVATE compiler_warnings
    Vulkan::Vulkan
    CONAN_PKG::fmt CONAN_PKG::glfw CONAN_PKG::glm
    )

set_target_properties(VulkanRenderer PROPERTIES RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/bin")


include(CompileShader)
compile_shader(vertShader
   SOURCE ${CMAKE_SOURCE_DIR}/data/shaders/shader.vert
   TARGET ${CMAKE_BINARY_DIR}/bin/shaders/shader.vert.spv
)
compile_shader(fragShader
   SOURCE ${CMAKE_SOURCE_DIR}/data/shaders/shader.frag
   TARGET ${CMAKE_BINARY_DIR}/bin/shaders/shader.frag.spv
)

add_dependencies(VulkanRenderer vertShader)
add_dependencies(VulkanRenderer fragShader)
