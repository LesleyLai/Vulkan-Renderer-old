#include <vulkan/vulkan.hpp>

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

#include "platform.hpp"

constexpr std::array validation_layers = {
    "VK_LAYER_LUNARG_standard_validation"};

#ifdef NDEBUG
constexpr bool vk_enable_validation_layers = false;
#else
constexpr bool vk_enable_validation_layers = true;
#endif

constexpr std::array device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

struct SwapChainSupportDetails {
  vk::SurfaceCapabilitiesKHR capabilities;
  std::vector<vk::SurfaceFormatKHR> formats;
  std::vector<vk::PresentModeKHR> present_modes;
};

[[nodiscard]] auto query_swapchain_support(const vk::PhysicalDevice& device,
                                           const vk::SurfaceKHR& surface)
    -> SwapChainSupportDetails
{
  SwapChainSupportDetails details;
  details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
  details.formats = device.getSurfaceFormatsKHR(surface);
  details.present_modes = device.getSurfacePresentModesKHR(surface);
  return details;
}

[[nodiscard]] auto choose_swap_surface_format(
    const std::vector<vk::SurfaceFormatKHR>& available_formats)
    -> vk::SurfaceFormatKHR
{
  if (available_formats.size() == 1 &&
      available_formats[0].format == vk::Format::eUndefined) {
    return {vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
  }

  for (const auto& available_format : available_formats) {
    if (available_format.format == vk::Format::eB8G8R8A8Unorm &&
        available_format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
      return available_format;
    }
  }

  return available_formats[0];
}

[[nodiscard]] auto choose_swap_present_mode(
    const std::vector<vk::PresentModeKHR>& available_present_modes)
    -> vk::PresentModeKHR
{
  vk::PresentModeKHR best_mode = vk::PresentModeKHR::eFifo;

  for (const auto& available_present_mode : available_present_modes) {
    if (available_present_mode == vk::PresentModeKHR::eMailbox) {
      return available_present_mode;
    } else if (available_present_mode == vk::PresentModeKHR::eImmediate) {
      best_mode = available_present_mode;
    }
  }

  return best_mode;
}

[[nodiscard]] auto
choose_swap_extent(const vk::SurfaceCapabilitiesKHR& capabilities,
                   const Platform& platform) -> vk::Extent2D
{
  if (capabilities.currentExtent.width !=
      std::numeric_limits<std::uint32_t>::max()) {
    return capabilities.currentExtent;
  } else {
    const auto res = platform.get_resolution();
    VkExtent2D actual_extent{static_cast<std::uint32_t>(res.width),
                             static_cast<std::uint32_t>(res.height)};

    actual_extent.width = std::max(
        capabilities.minImageExtent.width,
        std::min(capabilities.maxImageExtent.width, actual_extent.width));
    actual_extent.height = std::max(
        capabilities.minImageExtent.height,
        std::min(capabilities.maxImageExtent.height, actual_extent.height));

    return actual_extent;
  }
}

struct QueueFamilyIndices {
  std::optional<uint32_t> graphics_family;
  std::optional<uint32_t> present_family;

  bool is_complete()
  {
    return graphics_family.has_value() && present_family.has_value();
  }
};

static VKAPI_ATTR auto VKAPI_CALL vk_debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
    VkDebugUtilsMessageTypeFlagsEXT /*messageType*/,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void *
    /*pUserData*/) -> VkBool32
{
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
  return VK_FALSE;
}

class Application {
public:
  Application() : instance_{create_instance()}, dldy_{create_dynamic_loader()}
  {
    setup_debug_messenger();
    surface_ = platform_.create_vulkan_surface(instance_.get(), dldy_);
    pick_physical_device();
    create_logical_device();
    create_swap_chain();
  }

  void exec()
  {
    while (!platform_.should_close()) {
      platform_.poll_events();
    }
  }

private:
  Platform platform_;
  vk::UniqueInstance instance_;
  vk::DispatchLoaderDynamic dldy_;
  vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic>
      debug_messenger_;
  vk::UniqueHandle<vk::SurfaceKHR, vk::DispatchLoaderDynamic> surface_;
  vk::PhysicalDevice physical_device_ = nullptr;
  vk::UniqueDevice device_;

  vk::Queue graphics_queue_;
  vk::Queue present_queue_;
  vk::UniqueSwapchainKHR swapchain_;
  std::vector<vk::Image> swapchain_images_;
  vk::Format swapchain_image_format_;
  vk::Extent2D swapchain_extent_;

  [[nodiscard]] auto create_instance() -> vk::UniqueInstance
  {
    if (vk_enable_validation_layers) {
      check_validation_layer_support();
    }

    vk::ApplicationInfo app_info;
    app_info.setApiVersion(VK_API_VERSION_1_1);

    const auto extensions = get_required_extensions();

    vk::InstanceCreateInfo create_info;
    create_info.setPApplicationInfo(&app_info)
        .setEnabledExtensionCount(static_cast<std::uint32_t>(extensions.size()))
        .setPpEnabledExtensionNames(extensions.data());
    if (vk_enable_validation_layers) {
      create_info.setEnabledLayerCount(validation_layers.size())
          .setPpEnabledLayerNames(validation_layers.data());
    } else {
      create_info.setEnabledLayerCount(0);
    }

    return vk::createInstanceUnique(create_info);
  }

  [[nodiscard]] auto create_dynamic_loader() -> vk::DispatchLoaderDynamic
  {
    vk::DispatchLoaderDynamic dldy;
    dldy.init(instance_.get());
    return dldy;
  }

  void setup_debug_messenger()
  {
    if (!vk_enable_validation_layers) {
      return;
    }

    vk::DebugUtilsMessengerCreateInfoEXT create_info;
    create_info
        .setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
                            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                            vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose)
        .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation)
        .setPfnUserCallback(vk_debug_callback);

    debug_messenger_ = instance_->createDebugUtilsMessengerEXTUnique(
        create_info, nullptr, dldy_);
  }

  void pick_physical_device()
  {
    const auto devices = instance_->enumeratePhysicalDevices();
    assert(!devices.empty());

    for (const auto& device : devices) {
      if (is_physical_device_suitable(device)) {
        physical_device_ = device;
        break;
      }
    }

    assert(physical_device_);
  }

  void create_logical_device()
  {
    QueueFamilyIndices indices = find_queue_families(physical_device_);

    std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
    std::set<uint32_t> unique_queue_families = {indices.graphics_family.value(),
                                                indices.present_family.value()};

    float queue_priority = 1.0f;
    for (std::uint32_t queue_family : unique_queue_families) {
      vk::DeviceQueueCreateInfo create_info;
      create_info.setQueueFamilyIndex(queue_family)
          .setQueueCount(1)
          .setPQueuePriorities(&queue_priority);
      queue_create_infos.push_back(create_info);
    }

    vk::PhysicalDeviceFeatures device_features;

    vk::DeviceCreateInfo create_info;
    create_info.setPQueueCreateInfos(queue_create_infos.data())
        .setQueueCreateInfoCount(
            static_cast<uint32_t>(queue_create_infos.size()))
        .setPEnabledFeatures(&device_features)
        .setEnabledExtensionCount(
            static_cast<uint32_t>(device_extensions.size()))
        .setPpEnabledExtensionNames(device_extensions.data());

    if (vk_enable_validation_layers) {
      create_info.setEnabledLayerCount(validation_layers.size())
          .setPpEnabledLayerNames(validation_layers.data());
    } else {
      create_info.setEnabledLayerCount(0);
    }

    device_ = physical_device_.createDeviceUnique(create_info);
    graphics_queue_ = device_->getQueue(indices.graphics_family.value(), 0);
    present_queue_ = device_->getQueue(indices.present_family.value(), 0);
  }

  void create_swap_chain()
  {
    const auto swap_chain_support =
        query_swapchain_support(physical_device_, surface_.get());
    const auto surface_format =
        choose_swap_surface_format(swap_chain_support.formats);
    const auto present_mode =
        choose_swap_present_mode(swap_chain_support.present_modes);
    const auto extent =
        choose_swap_extent(swap_chain_support.capabilities, platform_);

    std::uint32_t image_count =
        swap_chain_support.capabilities.minImageCount + 1;
    if (swap_chain_support.capabilities.maxImageCount > 0 &&
        image_count > swap_chain_support.capabilities.maxImageCount) {
      image_count = swap_chain_support.capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR create_info;
    create_info.setSurface(surface_.get())
        .setMinImageCount(image_count)
        .setImageFormat(surface_format.format)
        .setImageColorSpace(surface_format.colorSpace)
        .setImageExtent(extent)
        .setImageArrayLayers(1)
        .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment);

    const QueueFamilyIndices indices = find_queue_families(physical_device_);
    std::array queue_family_indices = {indices.graphics_family.value(),
                                       indices.present_family.value()};

    if (indices.graphics_family != indices.present_family) {
      create_info.setImageSharingMode(vk::SharingMode::eConcurrent)
          .setQueueFamilyIndexCount(
              static_cast<std::uint32_t>(queue_family_indices.size()))
          .setPQueueFamilyIndices(queue_family_indices.data());
    } else {
      create_info.setImageSharingMode(vk::SharingMode::eExclusive);
    }

    create_info
        .setPreTransform(swap_chain_support.capabilities.currentTransform)
        .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
        .setPresentMode(present_mode)
        .setClipped(true)
        .setOldSwapchain(nullptr);

    swapchain_ = device_->createSwapchainKHRUnique(create_info);
    swapchain_images_ = device_->getSwapchainImagesKHR(swapchain_.get());
    swapchain_image_format_ = surface_format.format;
    swapchain_extent_ = extent;
  }

  [[nodiscard]] auto
  check_device_extension_support(const vk::PhysicalDevice& device)
  {
    const auto available_extensions =
        device.enumerateDeviceExtensionProperties();

    std::set<std::string> required_extensions(device_extensions.begin(),
                                              device_extensions.end());

    for (const auto& extension : available_extensions) {
      required_extensions.erase(
          static_cast<const char*>(extension.extensionName));
    }

    return required_extensions.empty();
  }

  [[nodiscard]] auto
  is_physical_device_suitable(const vk::PhysicalDevice& device) -> bool
  {
    QueueFamilyIndices indices = find_queue_families(device);

    const bool extensions_supported = check_device_extension_support(device);

    bool swap_chain_adequate = false;
    if (extensions_supported) {
      SwapChainSupportDetails swapChainSupport =
          query_swapchain_support(device, surface_.get());
      swap_chain_adequate = !swapChainSupport.formats.empty() &&
                            !swapChainSupport.present_modes.empty();
    }

    return indices.is_complete() && extensions_supported && swap_chain_adequate;
  }

  [[nodiscard]] auto find_queue_families(const vk::PhysicalDevice& device)
      -> QueueFamilyIndices
  {
    QueueFamilyIndices indices;
    const auto queue_family_properties = device.getQueueFamilyProperties();

    unsigned int i = 0;
    for (const auto& property : queue_family_properties) {
      if (property.queueCount > 0 &&
          property.queueFlags & vk::QueueFlagBits::eGraphics) {
        indices.graphics_family = i;
      }

      vk::Bool32 present_support = false;
      device.getSurfaceSupportKHR(i, surface_.get(), &present_support, dldy_);

      if (property.queueCount > 0 && present_support) {
        indices.present_family = i;
      }

      if (indices.is_complete()) {
        break;
      }

      ++i;
    }

    return indices;
  }

  [[nodiscard]] auto get_required_extensions() -> std::vector<const char*>
  {
    std::vector<const char*> extensions =
        platform_.get_required_vulkan_extensions();

    if (vk_enable_validation_layers) {
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
  }

  auto check_validation_layer_support() -> bool
  {
    const auto avaiable_layers = vk::enumerateInstanceLayerProperties();
    bool result = true;

    for (const char* layerName : validation_layers) {
      bool layer_found = false;

      for (const auto& layer_properties : avaiable_layers) {
        if (strcmp(layerName,
                   static_cast<const char*>(layer_properties.layerName)) == 0) {
          layer_found = true;
          break;
        }
      }

      if (!layer_found) {
        std::cerr << "Require Validation layer (" << layerName << ") no found";
        result = false;
      }
    }

    return result;
  }
};

int main()
{
  Application app;
  app.exec();
}
