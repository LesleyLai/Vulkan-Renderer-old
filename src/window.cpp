#include "window.hpp"

#include <GLFW/glfw3.h>

Window::Window(int width, int height, std::string_view name)
{
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window_ = glfwCreateWindow(width, height, name.data(), nullptr, nullptr);
}

Window::~Window()
{
  glfwDestroyWindow(window_);
  glfwTerminate();
}

Window::Window(Window&& other) noexcept : window_{other.window_}
{
  other.window_ = nullptr;
}

auto Window::operator=(Window&& other) noexcept -> Window&
{
  std::swap(window_, other.window_);
  return *this;
}

[[nodiscard]] auto Window::should_close() noexcept -> bool
{
  return glfwWindowShouldClose(window_);
}

void Window::poll_events() noexcept
{
  glfwPollEvents();
}

[[nodiscard]] auto Window::get_resolution() const noexcept -> Resolution
{
  int width, height;
  glfwGetWindowSize(window_, &width, &height);
  return Resolution{width, height};
}

[[nodiscard]] auto
Window::create_vulkan_surface(const vk::Instance& instance,
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

[[nodiscard]] auto Window::get_required_vulkan_extensions()
    -> std::vector<const char*>
{
  uint32_t glfw_extension_count = 0;
  const char** glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

  std::vector<const char*> extensions(glfw_extensions,
                                      glfw_extensions + glfw_extension_count);
  return extensions;
}
