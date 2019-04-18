find_package(Vulkan)

add_executable(VulkanRenderer "main.cpp" "platform.hpp" "platform.cpp")
target_link_libraries(VulkanRenderer
    PRIVATE compiler_warnings
    Vulkan::Vulkan
    glfw ${GLFW_LIBRARIES}
    )
