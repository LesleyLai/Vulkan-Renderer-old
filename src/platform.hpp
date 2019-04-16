#ifndef PLATFORM_HPP
#define PLATFORM_HPP

#include <vector>

struct GLFWwindow;

class Platform {
public:
  Platform();
  ~Platform();

  Platform(const Platform&) = delete;
  auto operator=(const Platform&) -> Platform& = delete;

  Platform(Platform&& other) noexcept;
  auto operator=(Platform&& other) noexcept -> Platform&;

  [[nodiscard]] auto should_close() -> bool;

  void poll_events();

  /**
   * @brief Returns the Vulkan instance extensions required by the platform.
   */
  [[nodiscard]] auto get_required_vulkan_extensions()
      -> std::vector<const char*>;

private:
  GLFWwindow* window_ = nullptr;
};

#endif // PLATFORM_HPP
