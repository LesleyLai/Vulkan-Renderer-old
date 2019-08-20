#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>
#include <fmt/format.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

#include "buffer_utils.hpp"
#include "camera.hpp"
#include "gltf.hpp"
#include "graphics_pipeline.hpp"
#include "shader_module.hpp"
#include "window.hpp"

constexpr std::array validation_layers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool vk_enable_validation_layers = false;
#else
constexpr bool vk_enable_validation_layers = true;
#endif

constexpr std::array device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

constexpr std::size_t frames_in_flight = 2;

constexpr vk::Format depth_format = vk::Format::eD32Sfloat;

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 texCoord;

  // Returns the vulkan binding description of a vertex
  [[nodiscard]] static auto binding_description()
      -> vk::VertexInputBindingDescription
  {
    return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
  }

  [[nodiscard]] static auto attributes_descriptions()
      -> std::vector<vk::VertexInputAttributeDescription>
  {
    return {
        vk::VertexInputAttributeDescription{0, 0, vk::Format::eR32G32B32Sfloat,
                                            offsetof(Vertex, pos)},
        vk::VertexInputAttributeDescription{1, 0, vk::Format::eR32G32B32Sfloat,
                                            offsetof(Vertex, color)},
        vk::VertexInputAttributeDescription{2, 0, vk::Format::eR32G32Sfloat,
                                            offsetof(Vertex, texCoord)},
    };
  }
};

const std::array vertices = {
    Vertex{{-0.5F, -0.5F, 0.0F}, {1.0F, 0.0F, 0.0F}, {1.0f, 0.0f}},
    Vertex{{0.5f, -0.5F, 0.0F}, {0.0F, 1.0F, 0.0F}, {0.0f, 0.0f}},
    Vertex{{0.5F, 0.5F, 0.0F}, {0.0F, 0.0F, 1.0F}, {0.0f, 1.0f}},
    Vertex{{-0.5F, 0.5F, 0.0F}, {1.0F, 1.0F, 1.0F}, {1.0f, 1.0f}},

    Vertex{{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
    Vertex{{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {2.0f, 0.0f}},
    Vertex{{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {2.0f, 2.0f}},
    Vertex{{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 2.0f}}};

const std::array<uint16_t, 12> indices{0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4};

struct UniformBufferObject {
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
};

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
    }
    if (available_present_mode == vk::PresentModeKHR::eImmediate) {
      best_mode = available_present_mode;
    }
  }

  return best_mode;
}

[[nodiscard]] auto
choose_swap_extent(const vk::SurfaceCapabilitiesKHR& capabilities,
                   const Window& window) -> vk::Extent2D
{
  if (capabilities.currentExtent.width !=
      std::numeric_limits<std::uint32_t>::max()) {
    return capabilities.currentExtent;
  }
  const auto res = window.get_resolution();
  VkExtent2D actual_extent{static_cast<std::uint32_t>(res.width),
                           static_cast<std::uint32_t>(res.height)};

  actual_extent.width =
      std::clamp(actual_extent.width, capabilities.minImageExtent.width,
                 capabilities.maxImageExtent.width);

  actual_extent.height =
      std::clamp(actual_extent.height, capabilities.minImageExtent.height,
                 capabilities.maxImageExtent.height);

  return actual_extent;
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
  fmt::print(stderr, "Validation layer: {}\n", pCallbackData->pMessage);
  std::fflush(stderr);
  return VK_FALSE;
}

static void framebuffer_resize_callback(GLFWwindow* window, int width,
                                        int height);

class Application {
public:
  bool frame_buffer_resized = false;

  Application()
      : window_{1440, 900, "Vulkan Renderer"}, instance_{create_instance()},
        dldy_{create_dynamic_loader()}
  {
    glfwSetFramebufferSizeCallback(window_.window(),
                                   framebuffer_resize_callback);
    glfwSetWindowUserPointer(window_.window(), this);

    debug_messenger_ = setup_debug_messenger();
    surface_ = window_.create_vulkan_surface(instance_.get(), dldy_);
    physical_device_ = pick_physical_device();
    queue_family_indices_ = find_queue_families(physical_device_);
    device_ = create_logical_device();
    graphics_queue_ =
        device_->getQueue(queue_family_indices_.graphics_family.value(), 0);
    present_queue_ =
        device_->getQueue(queue_family_indices_.present_family.value(), 0);

    create_swap_chain();
    create_swapchain_image_views();
    create_render_pass();
    create_descriptor_set_layout();

    vertex_shader_ =
        vulkan::create_shader_module("shaders/shader.vert.spv", *device_);
    frag_shader_ =
        vulkan::create_shader_module("shaders/shader.frag.spv", *device_);

    pipeline_layout_ = vulkan::create_graphics_pipeline_layout(
        *device_, *descriptor_set_layout_);

    create_graphics_pipelines();
    create_command_pool();
    create_depth_resource();
    create_frame_buffers();
    create_texture_image();
    create_texture_image_view();
    create_texture_sampler();
    load_model();
    create_vertex_buffer();
    create_index_buffer();
    create_uniform_buffers();
    create_descriptor_pool();
    create_descriptor_sets();
    create_command_buffers();
    create_sync_objects();
  }

  ~Application() = default;
  Application(const Application&) = delete;
  Application& operator=(const Application&) = delete;
  Application(Application&&) = delete;
  Application& operator=(Application&&) = delete;

  void exec()
  {
    while (!window_.should_close()) {
      window_.poll_events();
      render();
    }

    device_->waitIdle();
  }

private:
  Window window_;
  vk::UniqueInstance instance_;
  vk::DispatchLoaderDynamic dldy_;
  vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic>
      debug_messenger_;
  vk::UniqueHandle<vk::SurfaceKHR, vk::DispatchLoaderDynamic> surface_;
  vk::PhysicalDevice physical_device_;
  vk::UniqueDevice device_;

  QueueFamilyIndices queue_family_indices_;
  vk::Queue graphics_queue_;
  vk::Queue present_queue_;
  vk::UniqueSwapchainKHR swapchain_;
  std::vector<vk::Image> swapchain_images_;
  vk::Format swapchain_image_format_;
  vk::Extent2D swapchain_extent_;
  std::vector<vk::UniqueImageView> swapchain_image_views_;

  vk::UniqueImage depth_image_;
  vk::UniqueDeviceMemory depth_image_memory_;
  vk::UniqueImageView depth_image_view_;

  vk::UniqueRenderPass render_pass_;

  vk::UniqueShaderModule vertex_shader_;
  vk::UniqueShaderModule frag_shader_;

  vk::UniqueDescriptorSetLayout descriptor_set_layout_;
  vk::UniquePipelineLayout pipeline_layout_;
  vk::UniquePipeline graphics_pipeline_;

  std::vector<vk::UniqueFramebuffer> swapchain_framebuffers_;

  vk::UniqueDescriptorPool descriptor_pool_;
  std::vector<vk::DescriptorSet> descriptor_sets_;

  vk::UniqueCommandPool command_pool_;
  std::vector<vk::CommandBuffer> command_buffers_;

  std::array<vk::UniqueSemaphore, 2> image_available_semaphores;
  std::array<vk::UniqueSemaphore, 2> render_finished_semaphores;
  std::array<vk::UniqueFence, 2> in_flight_fences;
  size_t current_frame = 0;

  vk::UniqueBuffer vertex_buffer_;
  vk::UniqueDeviceMemory vertex_buffer_memory_;

  vk::UniqueBuffer index_buffer_;
  vk::UniqueDeviceMemory index_buffer_memory_;

  std::vector<vk::UniqueBuffer> uniform_buffers_;
  std::vector<vk::UniqueDeviceMemory> uniform_buffers_memory_;

  vk::UniqueImage texture_image_;
  vk::UniqueDeviceMemory texture_image_memory_;
  vk::UniqueImageView texture_image_view_;
  vk::UniqueSampler texture_sampler_;

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
      create_info
          .setEnabledLayerCount(static_cast<uint32_t>(validation_layers.size()))
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

  [[nodiscard]] auto setup_debug_messenger()
      -> vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic>
  {
    if constexpr (!vk_enable_validation_layers) {
      return vk::UniqueHandle<vk::DebugUtilsMessengerEXT,
                              vk::DispatchLoaderDynamic>{};
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

    return instance_->createDebugUtilsMessengerEXTUnique(create_info, nullptr,
                                                         dldy_);
  }

  [[nodiscard]] auto pick_physical_device() -> vk::PhysicalDevice
  {
    const auto devices = instance_->enumeratePhysicalDevices();
    assert(!devices.empty());

    for (const auto& device : devices) {
      if (is_physical_device_suitable(device)) {
        return device;
      }
    }

    std::fputs("Cannot find suitable physical device for Vulkan\n", stderr);
    return nullptr;
  }

  auto create_logical_device() -> vk::UniqueDevice
  {
    std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
    std::set<uint32_t> unique_queue_families = {
        queue_family_indices_.graphics_family.value(),
        queue_family_indices_.present_family.value()};

    const auto queue_family_properties =
        physical_device_.getQueueFamilyProperties();

    std::vector<std::vector<float>> queues_priorities;

    for (std::uint32_t queue_family : unique_queue_families) {
      const auto queue_count = queue_family_properties[queue_family].queueCount;

      queues_priorities.emplace_back(queue_count, 1.F);

      vk::DeviceQueueCreateInfo create_info;
      create_info.setQueueFamilyIndex(queue_family)
          .setQueueCount(queue_count)
          .setPQueuePriorities(queues_priorities.back().data());
      queue_create_infos.push_back(create_info);
    }

    vk::PhysicalDeviceFeatures device_features;
    device_features.samplerAnisotropy = true;

    vk::DeviceCreateInfo create_info;
    create_info.setPQueueCreateInfos(queue_create_infos.data())
        .setQueueCreateInfoCount(
            static_cast<uint32_t>(queue_create_infos.size()))
        .setPEnabledFeatures(&device_features)
        .setEnabledExtensionCount(
            static_cast<uint32_t>(device_extensions.size()))
        .setPpEnabledExtensionNames(device_extensions.data());

    if (vk_enable_validation_layers) {
      create_info
          .setEnabledLayerCount(static_cast<uint32_t>(validation_layers.size()))
          .setPpEnabledLayerNames(validation_layers.data());
    } else {
      create_info.setEnabledLayerCount(0);
    }

    return physical_device_.createDeviceUnique(create_info);
  }

  auto create_swap_chain() -> void
  {
    const auto swap_chain_support =
        query_swapchain_support(physical_device_, surface_.get());
    const auto surface_format =
        choose_swap_surface_format(swap_chain_support.formats);
    const auto present_mode =
        choose_swap_present_mode(swap_chain_support.present_modes);
    const auto extent =
        choose_swap_extent(swap_chain_support.capabilities, window_);

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

    std::array queue_family_indices = {
        queue_family_indices_.graphics_family.value(),
        queue_family_indices_.present_family.value()};

    if (queue_family_indices_.graphics_family !=
        queue_family_indices_.present_family) {
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

  auto create_swapchain_image_views() -> void
  {
    swapchain_image_views_.clear();
    swapchain_image_views_.reserve(swapchain_images_.size());

    for (const auto& image : swapchain_images_) {
      swapchain_image_views_.push_back(
          vulkan::create_image_view(*device_, image, swapchain_image_format_,
                                    vk::ImageAspectFlagBits::eColor));
    }
  }

  auto create_render_pass() -> void
  {
    vk::AttachmentDescription color_attachment;
    color_attachment.setFormat(swapchain_image_format_)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

    vk::AttachmentReference color_attachment_ref;
    color_attachment_ref.setAttachment(0).setLayout(
        vk::ImageLayout::eColorAttachmentOptimal);

    vk::AttachmentDescription depth_attachment;
    depth_attachment.setFormat(depth_format)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setInitialLayout(vk::ImageLayout::eUndefined)
        .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

    vk::AttachmentReference depth_attachment_ref;
    depth_attachment_ref.setAttachment(1).setLayout(
        vk::ImageLayout::eDepthStencilAttachmentOptimal);

    vk::SubpassDescription subpass;
    subpass.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
        .setColorAttachmentCount(1)
        .setPColorAttachments(&color_attachment_ref)
        .setPDepthStencilAttachment(&depth_attachment_ref);

    vk::SubpassDependency dependency{};
    dependency.setSrcSubpass(VK_SUBPASS_EXTERNAL)
        .setDstSubpass(0)
        .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
        .setSrcAccessMask(vk::AccessFlags{})
        .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
        .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead |
                          vk::AccessFlagBits::eColorAttachmentWrite);

    std::array attachments{color_attachment, depth_attachment};
    vk::RenderPassCreateInfo render_pass_create_info;
    render_pass_create_info
        .setAttachmentCount(static_cast<std::uint32_t>(attachments.size()))
        .setPAttachments(attachments.data())
        .setSubpassCount(1)
        .setPSubpasses(&subpass)
        .setDependencyCount(1)
        .setPDependencies(&dependency);

    render_pass_ = device_->createRenderPassUnique(render_pass_create_info);
  }

  auto create_descriptor_set_layout() -> void
  {
    const vk::DescriptorSetLayoutBinding ubo_layout_binding{
        0, vk::DescriptorType::eUniformBuffer, 1,
        vk::ShaderStageFlagBits::eVertex, nullptr};

    const vk::DescriptorSetLayoutBinding sampler_layout_binding{
        1, vk::DescriptorType::eCombinedImageSampler, 1,
        vk::ShaderStageFlagBits::eFragment, nullptr};

    std::array bindings = {ubo_layout_binding, sampler_layout_binding};

    const vk::DescriptorSetLayoutCreateInfo create_info{
        {}, static_cast<std::uint32_t>(bindings.size()), bindings.data()};

    descriptor_set_layout_ =
        device_->createDescriptorSetLayoutUnique(create_info);
  }

  auto create_graphics_pipelines() -> void
  {
    const vk::Viewport viewport{
        0,                                             // x
        static_cast<float>(swapchain_extent_.height),  // y
        static_cast<float>(swapchain_extent_.width),   // width
        -static_cast<float>(swapchain_extent_.height), // height
        0,                                             // minDepth
        1};                                            // maxDepth

    // Draw to the entire framebuffer
    const vk::Rect2D scissor{vk::Offset2D{0, 0}, swapchain_extent_};

    const vulkan::VertexInputInfo vertex_input_info{
        Vertex::binding_description(), Vertex::attributes_descriptions()};

    graphics_pipeline_ = vulkan::create_graphics_pipeline(
        *device_, *render_pass_, vk::PrimitiveTopology::eTriangleList,
        *pipeline_layout_, viewport, scissor,
        {.vertex = *vertex_shader_, .fragment = *frag_shader_, .tess = {}},
        vertex_input_info);
  }

  auto create_frame_buffers() -> void
  {
    swapchain_framebuffers_.clear();
    swapchain_framebuffers_.reserve(swapchain_image_views_.size());
    for (const auto& image_view : swapchain_image_views_) {
      std::array attachments{*image_view, *depth_image_view_};

      vk::FramebufferCreateInfo create_info;
      create_info.setRenderPass(*render_pass_)
          .setAttachmentCount(static_cast<uint32_t>(attachments.size()))
          .setPAttachments(attachments.data())
          .setWidth(swapchain_extent_.width)
          .setHeight(swapchain_extent_.height)
          .setLayers(1);

      swapchain_framebuffers_.emplace_back(
          device_->createFramebufferUnique(create_info));
    }
  }

  auto create_command_pool() -> void
  {
    const QueueFamilyIndices queue_family_indices =
        find_queue_families(physical_device_);

    const vk::CommandPoolCreateInfo create_info{
        {}, queue_family_indices.graphics_family.value()};

    command_pool_ = device_->createCommandPoolUnique(create_info);
  }

  auto find_memory_type(uint32_t type_filter,
                        const vk::MemoryPropertyFlags& properties)
      -> std::uint32_t
  {
    const auto device_memory_properties =
        physical_device_.getMemoryProperties();

    for (uint32_t i = 0; i < device_memory_properties.memoryTypeCount; i++) {
      if (type_filter & (1 << i) &&
          (device_memory_properties.memoryTypes[i].propertyFlags &
           properties) == properties) {
        return i;
      }
    }

    throw std::runtime_error("failed to find suitable memory type!");
  }

  auto create_depth_resource() -> void
  {
    const auto format = depth_format;
    std::tie(depth_image_, depth_image_memory_) = vulkan::create_image(
        physical_device_, *device_, swapchain_extent_.width,
        swapchain_extent_.height, format, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    depth_image_view_ = vulkan::create_image_view(
        *device_, *depth_image_, format, vk::ImageAspectFlagBits::eDepth);

    vulkan::transition_image_layout(
        *device_, graphics_queue_, *command_pool_, *depth_image_, format,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthStencilAttachmentOptimal);
  }

  auto create_texture_image() -> void
  {
    int tex_width, tex_height, tex_channels;
    stbi_uc* pixels = stbi_load("textures/texture.jpg", &tex_width, &tex_height,
                                &tex_channels, STBI_rgb_alpha);
    if (!pixels) {
      throw std::runtime_error("failed to load texture image!");
    }

    const auto image_size =
        static_cast<vk::DeviceSize>(tex_width * tex_height * 4);

    const auto [staging_buffer, staging_buffer_memory] =
        vulkan::create_buffer(physical_device_, *device_, image_size,
                              vk::BufferUsageFlagBits::eTransferSrc,
                              vk::MemoryPropertyFlagBits::eHostVisible |
                                  vk::MemoryPropertyFlagBits::eHostCoherent);

    void* data = device_->mapMemory(*staging_buffer_memory, 0, image_size);
    memcpy(data, pixels, static_cast<size_t>(image_size));
    device_->unmapMemory(*staging_buffer_memory);

    stbi_image_free(pixels);

    std::tie(texture_image_, texture_image_memory_) = vulkan::create_image(
        physical_device_, *device_, static_cast<std::uint32_t>(tex_width),
        static_cast<std::uint32_t>(tex_height), vk::Format::eR8G8B8A8Unorm,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    vulkan::transition_image_layout(*device_, graphics_queue_, *command_pool_,
                                    *texture_image_, vk::Format::eR8G8B8A8Unorm,
                                    vk::ImageLayout::eUndefined,
                                    vk::ImageLayout::eTransferDstOptimal);

    vulkan::copy_buffer_to_image(*device_, graphics_queue_, *command_pool_,
                                 *staging_buffer, *texture_image_,
                                 static_cast<std::uint32_t>(tex_width),
                                 static_cast<std::uint32_t>(tex_height));

    vulkan::transition_image_layout(*device_, graphics_queue_, *command_pool_,
                                    *texture_image_, vk::Format::eR8G8B8A8Unorm,
                                    vk::ImageLayout::eTransferDstOptimal,
                                    vk::ImageLayout::eShaderReadOnlyOptimal);
  }

  auto create_texture_image_view() -> void
  {
    texture_image_view_ = vulkan::create_image_view(
        *device_, *texture_image_, vk::Format::eR8G8B8A8Unorm,
        vk::ImageAspectFlagBits::eColor);
  }

  auto create_texture_sampler() -> void
  {
    vk::SamplerCreateInfo create_info;
    create_info.setMinFilter(vk::Filter::eLinear)
        .setMagFilter(vk::Filter::eLinear)
        .setAddressModeU(vk::SamplerAddressMode::eRepeat)
        .setAddressModeV(vk::SamplerAddressMode::eRepeat)
        .setAddressModeW(vk::SamplerAddressMode::eRepeat)
        .setAnisotropyEnable(true)
        .setMaxAnisotropy(16)
        .setBorderColor(vk::BorderColor::eIntOpaqueBlack)
        .setUnnormalizedCoordinates(false)
        .setCompareEnable(false)
        .setCompareOp(vk::CompareOp::eAlways)
        .setMipmapMode(vk::SamplerMipmapMode::eLinear)
        .setMipLodBias(0.f)
        .setMinLod(0.f)
        .setMaxLod(0.f);

    texture_sampler_ = device_->createSamplerUnique(create_info);
  }

  auto load_model() -> void
  {
    vulkan::Model model = vulkan::load_gltf_files("models/Box.gltf");
  }

  auto create_vertex_buffer() -> void
  {
    const auto size = sizeof(vertices[0]) * vertices.size();

    std::tie(vertex_buffer_, vertex_buffer_memory_) =
        vulkan::create_buffer_from_data(
            physical_device_, *device_, graphics_queue_, *command_pool_,
            vk::BufferUsageFlagBits::eVertexBuffer, vertices.data(), size);
  }

  auto create_index_buffer() -> void
  {
    const auto size = sizeof(indices[0]) * indices.size();
    std::tie(index_buffer_, index_buffer_memory_) =
        vulkan::create_buffer_from_data(
            physical_device_, *device_, graphics_queue_, *command_pool_,
            vk::BufferUsageFlagBits::eIndexBuffer, indices.data(), size);
  }

  auto create_descriptor_pool() -> void
  {
    std::array<vk::DescriptorPoolSize, 2> pool_sizes;
    pool_sizes[0]
        .setType(vk::DescriptorType::eUniformBuffer)
        .setDescriptorCount(static_cast<uint32_t>(swapchain_images_.size()));
    pool_sizes[1]
        .setType(vk::DescriptorType::eCombinedImageSampler)
        .setDescriptorCount(static_cast<uint32_t>(swapchain_images_.size()));

    const vk::DescriptorPoolCreateInfo create_info{
        {},
        static_cast<uint32_t>(swapchain_images_.size()),
        static_cast<uint32_t>(pool_sizes.size()),
        pool_sizes.data()};

    descriptor_pool_ = device_->createDescriptorPoolUnique(create_info);
  }

  auto create_descriptor_sets() -> void
  {
    std::vector<vk::DescriptorSetLayout> layouts(swapchain_images_.size(),
                                                 *descriptor_set_layout_);

    vk::DescriptorSetAllocateInfo alloc_info{
        *descriptor_pool_, static_cast<uint32_t>(layouts.size()),
        layouts.data()};

    descriptor_sets_ = device_->allocateDescriptorSets(alloc_info);

    for (std::size_t i = 0; i < descriptor_sets_.size(); ++i) {
      const vk::DescriptorBufferInfo buffer_info{*uniform_buffers_[i], 0,
                                                 VK_WHOLE_SIZE};

      const vk::DescriptorImageInfo image_info{
          *texture_sampler_, *texture_image_view_,
          vk::ImageLayout::eShaderReadOnlyOptimal};

      std::array writes{
          vk::WriteDescriptorSet{descriptor_sets_[i], 0, 0, 1,
                                 vk::DescriptorType::eUniformBuffer, nullptr,
                                 &buffer_info, nullptr},
          vk::WriteDescriptorSet{descriptor_sets_[i], 1, 0, 1,
                                 vk::DescriptorType::eCombinedImageSampler,
                                 &image_info, nullptr, nullptr}

      };

      device_->updateDescriptorSets(static_cast<uint32_t>(writes.size()),
                                    writes.data(), 0, nullptr);
    }
  }

  auto create_uniform_buffers() -> void
  {
    vk::DeviceSize buffer_size = sizeof(UniformBufferObject);

    const auto images_count = swapchain_images_.size();
    uniform_buffers_.resize(images_count);
    uniform_buffers_memory_.resize(images_count);

    for (std::size_t i = 0; i < images_count; ++i) {
      std::tie(uniform_buffers_[i], uniform_buffers_memory_[i]) =
          vulkan::create_buffer(physical_device_, *device_, buffer_size,
                                vk::BufferUsageFlagBits::eUniformBuffer,
                                vk::MemoryPropertyFlagBits::eHostVisible |
                                    vk::MemoryPropertyFlagBits::eHostCoherent);
    }
  }

  auto create_command_buffers() -> void
  {
    const auto command_buffers_count = swapchain_framebuffers_.size();

    vk::CommandBufferAllocateInfo alloc_info;
    alloc_info.setCommandPool(*command_pool_)
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(
            static_cast<std::uint32_t>(command_buffers_count));

    command_buffers_ = device_->allocateCommandBuffers(alloc_info);

    for (size_t i = 0; i < command_buffers_count; ++i) {
      const auto& command_buffer = command_buffers_[i];

      const vk::CommandBufferBeginInfo command_buffer_begin_info{
          vk::CommandBufferUsageFlagBits::eSimultaneousUse, nullptr};

      command_buffer.begin(&command_buffer_begin_info);

      vk::RenderPassBeginInfo render_pass_begin_info;
      render_pass_begin_info.setRenderPass(*render_pass_)
          .setFramebuffer(*swapchain_framebuffers_[i])
          .setRenderArea(vk::Rect2D{{0, 0}, swapchain_extent_});

      std::array<vk::ClearValue, 2> clear_values;
      clear_values[0].setColor(std::array{0.F, 0.F, 0.F, 1.F});
      clear_values[1].setDepthStencil({1.0F, 0});

      render_pass_begin_info
          .setClearValueCount(static_cast<std::uint32_t>(clear_values.size()))
          .setPClearValues(clear_values.data());

      command_buffer.beginRenderPass(&render_pass_begin_info,
                                     vk::SubpassContents::eInline);

      command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                  *graphics_pipeline_);

      vk::DeviceSize offset{0};
      command_buffer.bindVertexBuffers(0, 1, &vertex_buffer_.get(), &offset);
      command_buffer.bindIndexBuffer(*index_buffer_, 0, vk::IndexType::eUint16);

      command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                        *pipeline_layout_, 0, 1,
                                        &descriptor_sets_[i], 0, nullptr);
      command_buffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0,
                                 0);

      command_buffer.endRenderPass();

      command_buffer.end();
    }
  }

  auto create_sync_objects() -> void
  {
    const vk::SemaphoreCreateInfo semaphore_create_info;
    const vk::FenceCreateInfo fence_create_info{
        vk::FenceCreateFlagBits::eSignaled};
    for (size_t i = 0; i < frames_in_flight; ++i) {
      image_available_semaphores[i] =
          device_->createSemaphoreUnique(semaphore_create_info);
      render_finished_semaphores[i] =
          device_->createSemaphoreUnique(semaphore_create_info);
      in_flight_fences[i] = device_->createFenceUnique(fence_create_info);
    }
  }

  auto recreate_swapchain() -> void
  {
    frame_buffer_resized = false;

    Resolution res{0, 0};
    while (res.width == 0 && res.height == 0) {
      res = window_.get_resolution();
      glfwWaitEvents();
    }

    device_->waitIdle();

    swapchain_.reset();

    create_swap_chain();
    create_swapchain_image_views();
    create_render_pass();
    create_graphics_pipelines();
    create_depth_resource();
    create_frame_buffers();
    create_uniform_buffers();
    create_descriptor_pool();
    create_descriptor_sets();
    create_command_buffers();
  }

  auto update_uniform_buffer(std::uint32_t current_image) -> void
  {
    static auto start_time = std::chrono::high_resolution_clock::now();
    const auto current_time = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(
                     current_time - start_time)
                     .count();

    UniformBufferObject ubo = {};
    ubo.model = glm::rotate(glm::mat4(1.0F), time * glm::radians(90.0F),
                            glm::vec3(0.0F, 0.0F, 1.0F));
    ubo.view =
        glm::lookAt(glm::vec3(2.0F, 2.0F, 2.0F), glm::vec3(0.0F, 0.0F, 0.0F),
                    glm::vec3(0.0F, 0.0F, 1.0F));
    ubo.proj = glm::perspective(
        glm::radians(45.0F),
        swapchain_extent_.width / static_cast<float>(swapchain_extent_.height),
        0.1F, 10.0F);
    // ubo.proj[1][1] *= -1;

    void* data = device_->mapMemory(*uniform_buffers_memory_[current_image], 0,
                                    sizeof(ubo));
    memcpy(data, &ubo, sizeof(ubo));
    device_->unmapMemory(*uniform_buffers_memory_[current_image]);
  }

  auto render() -> void
  {
    device_->waitForFences(1, &(*in_flight_fences[current_frame]), true,
                           std::numeric_limits<uint64_t>::max());

    const auto [result, image_index] = device_->acquireNextImageKHR(
        *swapchain_, std::numeric_limits<uint64_t>::max(),
        *image_available_semaphores[current_frame], nullptr);

    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR) {
      recreate_swapchain();
      return;
    }
    assert(result == vk::Result::eSuccess);

    update_uniform_buffer(image_index);

    vk::SubmitInfo submit_info;
    const std::array wait_semaphores = {
        *image_available_semaphores[current_frame]};
    const std::array wait_stages{vk::PipelineStageFlags{
        vk::PipelineStageFlagBits::eColorAttachmentOutput}};
    const std::array signal_semaphores = {
        *render_finished_semaphores[current_frame]};

    submit_info
        .setWaitSemaphoreCount(
            static_cast<std::uint32_t>(wait_semaphores.size()))
        .setPWaitSemaphores(wait_semaphores.data())
        .setPWaitDstStageMask(wait_stages.data())
        .setCommandBufferCount(1)
        .setPCommandBuffers(&command_buffers_[image_index])
        .setSignalSemaphoreCount(
            static_cast<std::uint32_t>(signal_semaphores.size()))
        .setPSignalSemaphores(signal_semaphores.data());

    device_->resetFences(1, &*in_flight_fences[current_frame]);

    graphics_queue_.submit(1, &submit_info, *in_flight_fences[current_frame]);

    vk::PresentInfoKHR present_info;
    present_info
        .setWaitSemaphoreCount(
            static_cast<unsigned int>(signal_semaphores.size()))
        .setPWaitSemaphores(signal_semaphores.data());

    const std::array swap_chains{*swapchain_};
    present_info
        .setSwapchainCount(static_cast<unsigned int>(swap_chains.size()))
        .setPSwapchains(swap_chains.data())
        .setPImageIndices(&image_index);

    {
      const auto result2 = present_queue_.presentKHR(&present_info);
      if (result2 == vk::Result::eErrorOutOfDateKHR ||
          result2 == vk::Result::eSuboptimalKHR || frame_buffer_resized) {
        recreate_swapchain();
      }
    }

    current_frame = (current_frame + 1) % frames_in_flight;
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

    const auto supported_features = device.getFeatures();

    return indices.is_complete() && extensions_supported &&
           swap_chain_adequate && supported_features.samplerAnisotropy;
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
      device.getSurfaceSupportKHR(i, surface_.get(), &present_support);

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
        window_.get_required_vulkan_extensions();

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
        fmt::print(stderr, "Required Validation layer ({}) not found",
                   layerName);
        result = false;
      }
    }

    return result;
  }
};

static void framebuffer_resize_callback(GLFWwindow* window, int /*width*/,
                                        int /*height*/)
{
  auto app = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
  app->frame_buffer_resized = true;
}

int main()
try {
  Application app;
  app.exec();
} catch (const std::exception& e) {
  fmt::print(stderr, "Error: {}\n", e.what());
} catch (...) {
  std::fputs("Unknown exception thrown!\n", stderr);
}
