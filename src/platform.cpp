#include "platform.hpp"

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

Platform::Platform()
{
  glfwInit();
  window_ = glfwCreateWindow(1440, 900, "Vulkan", nullptr, nullptr);
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

[[nodiscard]] auto Platform::get_resolution() const noexcept -> Resolution
{
  int width, height;
  glfwGetWindowSize(window_, &width, &height);
  return Resolution{width, height};
}

[[nodiscard]] auto
Platform::create_vulkan_surface(const vk::Instance& instance) const
    -> vk::UniqueSurfaceKHR
{
#if defined(VK_USE_PLATFORM_WIN32_KHR)
  HWND window = glfwGetWin32Window(window_);
  vk::Win32SurfaceCreateInfoKHR create_info(vk::Win32SurfaceCreateFlagsKHR(),
                                            GetModuleHandle(nullptr), window);
  return instance.createWin32SurfaceKHRUnique(create_info);
#else
#pragma error "unhandled platform"
#endif
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
