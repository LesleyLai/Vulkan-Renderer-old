#ifndef PLATFORM_HPP
#define PLATFORM_HPP

#include <vector>

#include <vulkan/vulkan.hpp>

struct GLFWwindow;

struct Resolution {
  int width;
  int height;
};

class Platform {
public:
  Platform();
  ~Platform();

  Platform(const Platform&) = delete;
  auto operator=(const Platform&) -> Platform& = delete;

  Platform(Platform&& other) noexcept;
  auto operator=(Platform&& other) noexcept -> Platform&;

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

#endif // PLATFORM_HPP
