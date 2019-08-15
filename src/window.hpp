#ifndef WINDOW_HPP
#define WINDOW_HPP

#include <vector>
#include <string_view>

#include <vulkan/vulkan.hpp>

struct GLFWwindow;

struct Resolution {
  int width;
  int height;
};

class Window {
public:
  Window(int width, int height, std::string_view name);
  ~Window();

  Window(const Window&) = delete;
  auto operator=(const Window&) -> Window& = delete;

  Window(Window&& other) noexcept;
  auto operator=(Window&& other) noexcept -> Window&;

  [[nodiscard]] auto should_close() noexcept -> bool;

  void poll_events() noexcept;

  [[nodiscard]] auto get_resolution() const noexcept -> Resolution;

  /**
   * @brief Returns the Vulkan instance extensions required by the platform.
   */
  [[nodiscard]] auto get_required_vulkan_extensions()
      -> std::vector<const char*>;

  [[nodiscard]] auto
  create_vulkan_surface(const vk::Instance& instance,
                        const vk::DispatchLoaderDynamic& dldy) const
      -> vk::UniqueHandle<vk::SurfaceKHR, vk::DispatchLoaderDynamic>;

  [[nodiscard]] auto window() const noexcept -> GLFWwindow*
  {
    return window_;
  }

private:
  GLFWwindow* window_ = nullptr;
};

#endif // WINDOW_HPP
