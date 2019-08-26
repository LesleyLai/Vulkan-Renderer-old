find_package(Vulkan)

add_executable(VulkanRenderer "main.cpp"
    "buffer_utils.hpp" "buffer_utils.cpp"
    "camera.hpp"
    "graphics_pipeline.hpp" "graphics_pipeline.cpp"
    "shader_module.hpp" "shader_module.cpp"
    "window.hpp" "window.cpp")
target_link_libraries(VulkanRenderer
    PRIVATE compiler_warnings
    Vulkan::Vulkan
    CONAN_PKG::fmt CONAN_PKG::glfw CONAN_PKG::glm CONAN_PKG::stb
    )

set_target_properties(VulkanRenderer PROPERTIES RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/bin")

# Shaders
include(CompileShader)
compile_shader(vertShader
   SOURCE ${CMAKE_SOURCE_DIR}/shaders/shader.vert
   TARGET ${CMAKE_BINARY_DIR}/bin/shaders/shader.vert.spv
)

compile_shader(fragShader
   SOURCE ${CMAKE_SOURCE_DIR}/shaders/shader.frag
   TARGET ${CMAKE_BINARY_DIR}/bin/shaders/shader.frag.spv
)

target_compile_definitions(VulkanRenderer PUBLIC
    GLM_FORCE_RADIANS GLM_FORCE_DEPTH_ZERO_TO_ONE)

add_dependencies(VulkanRenderer vertShader)
add_dependencies(VulkanRenderer fragShader)

# Copy assets
add_custom_target(assets
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/data ${CMAKE_BINARY_DIR}/bin
        )

add_dependencies(VulkanRenderer assets)
