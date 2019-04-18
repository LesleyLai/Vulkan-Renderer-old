#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
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
    surface_ = platform_.create_vulkan_surface(instance_.get());
    pick_physical_device();
    create_logical_device();
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
  vk::UniqueSurfaceKHR surface_;
  vk::PhysicalDevice physical_device_ = nullptr;
  vk::UniqueDevice device_;

  vk::Queue graphics_queue_;

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

    const float queue_priority = 1.0f;
    vk::DeviceQueueCreateInfo queue_create_info;
    queue_create_info.setQueueFamilyIndex(indices.graphics_family.value())
        .setQueueCount(1)
        .setPQueuePriorities(&queue_priority);

    vk::PhysicalDeviceFeatures device_features;

    vk::DeviceCreateInfo create_info;
    create_info.setPQueueCreateInfos(&queue_create_info)
        .setQueueCreateInfoCount(1)
        .setPEnabledFeatures(&device_features)
        .setEnabledExtensionCount(0);

    if (vk_enable_validation_layers) {
      create_info.setEnabledLayerCount(validation_layers.size())
          .setPpEnabledLayerNames(validation_layers.data());
    } else {
      create_info.setEnabledLayerCount(0);
    }

    device_ = physical_device_.createDeviceUnique(create_info);
    device_->getQueue(indices.graphics_family.value(), 0, &graphics_queue_);
  }

  [[nodiscard]] auto is_physical_device_suitable(vk::PhysicalDevice device)
      -> bool
  {
    QueueFamilyIndices indices = find_queue_families(device);

    return indices.is_complete();
  }

  [[nodiscard]] auto find_queue_families(vk::PhysicalDevice device)
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
        if (strcmp(layerName, layer_properties.layerName) == 0) {
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
