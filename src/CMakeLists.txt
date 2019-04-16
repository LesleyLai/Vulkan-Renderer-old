add_executable(VulkanRenderer "main.cpp" "platform.hpp" "platform.cpp")
target_link_libraries(VulkanRenderer
    PRIVATE compiler_warnings
    glfw ${GLFW_LIBRARIES}
    vulkan
    )
