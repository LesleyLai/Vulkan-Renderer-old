#include "platform.hpp"

#include <GLFW/glfw3.h>

Platform::Platform()
{
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window_ = glfwCreateWindow(1440, 900, "Vulkan Renderer", nullptr, nullptr);
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

[[nodiscard]] auto Platform::should_close() noexcept -> bool
{
  return glfwWindowShouldClose(window_);
}

void Platform::poll_events() noexcept
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
Platform::create_vulkan_surface(const vk::Instance& instance,
                                const vk::DispatchLoaderDynamic& dldy) const
    -> vk::UniqueHandle<vk::SurfaceKHR, vk::DispatchLoaderDynamic>
{
  VkSurfaceKHR surface;
  glfwCreateWindowSurface(static_cast<VkInstance>(instance), window_, nullptr,
                          &surface);

  const vk::ObjectDestroy<vk::Instance, vk::DispatchLoaderDynamic> destroyer(
      instance, nullptr, dldy);
  return vk::UniqueHandle<vk::SurfaceKHR, vk::DispatchLoaderDynamic>{
      vk::SurfaceKHR{surface}, destroyer};
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
