#include "platform.hpp"
#include <GLFW/glfw3.h>

Platform::Platform()
{
  glfwInit();
  window_ = glfwCreateWindow(800, 600, "Vulkan", nullptr, nullptr);
}

Platform::~Platform()
{
  glfwDestroyWindow(window_);
  glfwTerminate();
}

Platform::Platform(Platform&& other) noexcept : window_{other.window_}
{
  other.window_ = nullptr;
}

auto Platform::operator=(Platform&& other) noexcept -> Platform&
{
  std::swap(window_, other.window_);
  return *this;
}

[[nodiscard]] auto Platform::should_close() -> bool
{
  return glfwWindowShouldClose(window_);
}

void Platform::poll_events()
{
  glfwPollEvents();
}

[[nodiscard]] auto Platform::get_required_vulkan_extensions()
    -> std::vector<const char*>
{
  uint32_t glfw_extension_count = 0;
  const char** glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

  std::vector<const char*> extensions(glfw_extensions,
                                      glfw_extensions + glfw_extension_count);
  return extensions;
}
