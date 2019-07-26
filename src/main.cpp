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

#include "platform.hpp"
#include "shader_module.hpp"

constexpr std::array validation_layers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool vk_enable_validation_layers = false;
#else
constexpr bool vk_enable_validation_layers = true;
#endif

constexpr std::array device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

constexpr std::size_t frames_in_flight = 2;

struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;

  // Returns the vulkan binding description of a vertex
  [[nodiscard]] static auto binding_description()
      -> vk::VertexInputBindingDescription
  {
    return {0, sizeof(Vertex), vk::VertexInputRate::eVertex};
  }

  [[nodiscard]] static auto attributes_descriptions()
      -> std::array<vk::VertexInputAttributeDescription, 2>
  {
    return {
        vk::VertexInputAttributeDescription{0, 0, vk::Format::eR32G32Sfloat,
                                            offsetof(Vertex, pos)},
        vk::VertexInputAttributeDescription{1, 0, vk::Format::eR32G32B32Sfloat,
                                            offsetof(Vertex, color)},
    };
  }
};

const std::array<Vertex, 4> vertices = {
    Vertex{{-0.5F, -0.5F}, {1.0F, 0.0F, 0.0F}},
    Vertex{{0.5f, -0.5F}, {0.0F, 1.0F, 0.0F}},
    Vertex{{0.5F, 0.5F}, {0.0F, 0.0F, 1.0F}},
    Vertex{{-0.5F, 0.5F}, {1.0F, 1.0F, 1.0F}}};

const std::array<uint16_t, 6> indices{0, 1, 2, 2, 3, 0};

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
                   const Platform& platform) -> vk::Extent2D
{
  if (capabilities.currentExtent.width !=
      std::numeric_limits<std::uint32_t>::max()) {
    return capabilities.currentExtent;
  }
  const auto res = platform.get_resolution();
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
      : platform_{1440, 900, "Vulkan Renderer"}, instance_{create_instance()},
        dldy_{create_dynamic_loader()}
  {
    glfwSetFramebufferSizeCallback(platform_.window(),
                                   framebuffer_resize_callback);
    glfwSetWindowUserPointer(platform_.window(), this);

    debug_messenger_ = setup_debug_messenger();
    surface_ = platform_.create_vulkan_surface(instance_.get(), dldy_);
    physical_device_ = pick_physical_device();
    queue_family_indices_ = find_queue_families(physical_device_);
    device_ = create_logical_device();
    graphics_queue_ =
        device_->getQueue(queue_family_indices_.graphics_family.value(), 0);
    present_queue_ =
        device_->getQueue(queue_family_indices_.present_family.value(), 0);

    create_swap_chain();
    create_image_views();
    create_render_pass();
    create_descriptor_set_layout();
    create_graphics_pipeline();
    create_frame_buffers();
    create_command_pool();
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
    while (!platform_.should_close()) {
      platform_.poll_events();
      render();
    }

    device_->waitIdle();
  }

private:
  Platform platform_;
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

  vk::UniqueRenderPass render_pass_;
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

  auto create_image_views() -> void
  {
    swapchain_image_views_.clear();
    swapchain_image_views_.reserve(swapchain_images_.size());

    for (const auto& image : swapchain_images_) {
      vk::ImageSubresourceRange subresource_range;
      subresource_range.setAspectMask(vk::ImageAspectFlagBits::eColor)
          .setBaseMipLevel(0)
          .setLevelCount(1)
          .setBaseArrayLayer(0)
          .setLayerCount(1);

      vk::ImageViewCreateInfo create_info;
      create_info.setImage(image)
          .setViewType(vk::ImageViewType::e2D)
          .setFormat(swapchain_image_format_)
          .setComponents(vk::ComponentMapping{})
          .setSubresourceRange(subresource_range);

      swapchain_image_views_.emplace_back(
          device_->createImageViewUnique(create_info));
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

    vk::SubpassDescription subpass;
    subpass.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
        .setColorAttachmentCount(1)
        .setPColorAttachments(&color_attachment_ref);

    vk::SubpassDependency dependency{};
    dependency.setSrcSubpass(VK_SUBPASS_EXTERNAL)
        .setDstSubpass(0)
        .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
        .setSrcAccessMask(vk::AccessFlags{})
        .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
        .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead |
                          vk::AccessFlagBits::eColorAttachmentWrite);

    vk::RenderPassCreateInfo render_pass_create_info;
    render_pass_create_info.setAttachmentCount(1)
        .setPAttachments(&color_attachment)
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

    const vk::DescriptorSetLayoutCreateInfo create_info{
        {}, 1, &ubo_layout_binding};

    descriptor_set_layout_ =
        device_->createDescriptorSetLayoutUnique(create_info);
  }

  auto create_graphics_pipeline() -> void
  {
    auto vert_shader_module =
        create_shader_module("shaders/shader.vert.spv", *device_);
    auto frag_shader_module =
        create_shader_module("shaders/shader.frag.spv", *device_);

    vk::PipelineShaderStageCreateInfo vert_shader_stage_info;
    vert_shader_stage_info.setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(*vert_shader_module)
        .setPName("main");
    vk::PipelineShaderStageCreateInfo frag_shader_stage_info;
    frag_shader_stage_info.setStage(vk::ShaderStageFlagBits::eFragment)
        .setModule(*frag_shader_module)
        .setPName("main");

    const auto binding_description = Vertex::binding_description();
    const auto attribute_descriptions = Vertex::attributes_descriptions();

    const vk::PipelineVertexInputStateCreateInfo vertex_input_stage_create_info{
        {},
        1,
        &binding_description,
        static_cast<uint32_t>(attribute_descriptions.size()),
        attribute_descriptions.data()};

    const vk::PipelineInputAssemblyStateCreateInfo input_assembly{
        {}, vk::PrimitiveTopology::eTriangleList, false};

    const vk::Viewport viewport{
        0,                                            // x
        0,                                            // y
        static_cast<float>(swapchain_extent_.width),  // width
        static_cast<float>(swapchain_extent_.height), // height
        0,                                            // minDepth
        1};                                           // maxDepth

    // Draw to the entire framebuffer
    const vk::Rect2D scissor{vk::Offset2D{0, 0}, swapchain_extent_};

    vk::PipelineViewportStateCreateInfo viewport_state_create_info;
    viewport_state_create_info.setViewportCount(1)
        .setPViewports(&viewport)
        .setScissorCount(1)
        .setPScissors(&scissor);

    vk::PipelineRasterizationStateCreateInfo rasterizer_create_info;
    rasterizer_create_info.setDepthClampEnable(false)
        .setRasterizerDiscardEnable(false)
        .setPolygonMode(vk::PolygonMode::eFill)
        .setLineWidth(1)
        .setCullMode(vk::CullModeFlagBits::eBack)
        .setFrontFace(vk::FrontFace::eClockwise)
        .setDepthBiasEnable(false);

    vk::PipelineMultisampleStateCreateInfo multisampling_create_info;
    multisampling_create_info.setSampleShadingEnable(false)
        .setRasterizationSamples(vk::SampleCountFlagBits::e1);

    vk::PipelineColorBlendAttachmentState color_blend_attachment;
    color_blend_attachment
        .setColorWriteMask(
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)
        .setBlendEnable(false);

    vk::PipelineColorBlendStateCreateInfo color_blend_create_info;
    color_blend_create_info.setLogicOpEnable(false)
        .setLogicOp(vk::LogicOp::eCopy)
        .setAttachmentCount(1)
        .setPAttachments(&color_blend_attachment)
        .setBlendConstants({0, 0, 0, 0});

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info;
    pipeline_layout_create_info.setSetLayoutCount(1)
        .setPSetLayouts(&descriptor_set_layout_.get())
        .setPushConstantRangeCount(0);

    pipeline_layout_ =
        device_->createPipelineLayoutUnique(pipeline_layout_create_info);

    std::array shader_stages{vert_shader_stage_info, frag_shader_stage_info};

    vk::GraphicsPipelineCreateInfo pipeline_create_info;
    pipeline_create_info
        .setStageCount(static_cast<std::uint32_t>(shader_stages.size()))
        .setPStages(shader_stages.data())
        .setPVertexInputState(&vertex_input_stage_create_info)
        .setPInputAssemblyState(&input_assembly)
        .setPViewportState(&viewport_state_create_info)
        .setPRasterizationState(&rasterizer_create_info)
        .setPMultisampleState(&multisampling_create_info)
        .setPColorBlendState(&color_blend_create_info)
        .setLayout(*pipeline_layout_)
        .setRenderPass(*render_pass_)
        .setSubpass(0)
        .setBasePipelineHandle(nullptr);

    graphics_pipeline_ =
        device_->createGraphicsPipelineUnique(nullptr, pipeline_create_info);
  }

  auto create_frame_buffers() -> void
  {
    swapchain_framebuffers_.clear();
    swapchain_framebuffers_.reserve(swapchain_image_views_.size());
    for (const auto& image_view : swapchain_image_views_) {
      std::array attachments{*image_view};

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

  auto copy_buffer(vk::Buffer src, vk::Buffer dst, vk::DeviceSize size) -> void
  {
    const vk::CommandBufferAllocateInfo alloc_info{
        *command_pool_, vk::CommandBufferLevel::ePrimary, 1};

    const auto command_buffers =
        device_->allocateCommandBuffersUnique(alloc_info);
    const auto& command_buffer = command_buffers[0].get();

    const vk::CommandBufferBeginInfo begin_info{
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
    command_buffer.begin(begin_info);
    vk::BufferCopy copy_region{0, 0, size};
    command_buffer.copyBuffer(src, dst, 1, &copy_region);
    command_buffer.end();

    vk::SubmitInfo submit_info;
    submit_info.setCommandBufferCount(1).setPCommandBuffers(&command_buffer);
    graphics_queue_.submit(1, &submit_info, vk::Fence{});
    graphics_queue_.waitIdle();
  }

  auto create_vertex_buffer() -> void
  {
    const auto size = sizeof(vertices[0]) * vertices.size();
    const auto [staging_buffer, staging_buffer_memory] =
        create_buffer(*device_, size, vk::BufferUsageFlagBits::eTransferSrc,
                      vk::MemoryPropertyFlagBits::eHostVisible |
                          vk::MemoryPropertyFlagBits::eHostCoherent);

    void* data;
    device_->mapMemory(*staging_buffer_memory, 0, size, {}, &data);
    memcpy(data, vertices.data(), size);
    device_->unmapMemory(*staging_buffer_memory);

    std::tie(vertex_buffer_, vertex_buffer_memory_) =
        create_buffer(*device_, size,
                      vk::BufferUsageFlagBits::eTransferDst |
                          vk::BufferUsageFlagBits::eVertexBuffer,
                      vk::MemoryPropertyFlagBits::eDeviceLocal);

    copy_buffer(*staging_buffer, *vertex_buffer_, size);
  }

  auto create_index_buffer() -> void
  {
    const auto size = sizeof(indices[0]) * indices.size();
    const auto [staging_buffer, staging_buffer_memory] =
        create_buffer(*device_, size, vk::BufferUsageFlagBits::eTransferSrc,
                      vk::MemoryPropertyFlagBits::eHostVisible |
                          vk::MemoryPropertyFlagBits::eHostCoherent);

    void* data;
    device_->mapMemory(*staging_buffer_memory, 0, size, {}, &data);
    memcpy(data, indices.data(), size);
    device_->unmapMemory(*staging_buffer_memory);

    std::tie(index_buffer_, index_buffer_memory_) =
        create_buffer(*device_, size,
                      vk::BufferUsageFlagBits::eTransferDst |
                          vk::BufferUsageFlagBits::eIndexBuffer,
                      vk::MemoryPropertyFlagBits::eDeviceLocal);

    copy_buffer(*staging_buffer, *index_buffer_, size);
  }

  auto create_descriptor_pool() -> void
  {
    const vk::DescriptorPoolSize pool_size{
        vk::DescriptorType::eUniformBuffer,
        static_cast<uint32_t>(swapchain_images_.size())};
    const vk::DescriptorPoolCreateInfo create_info{
        {}, static_cast<uint32_t>(swapchain_images_.size()), 1, &pool_size};

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

      const vk::WriteDescriptorSet write{descriptor_sets_[i],
                                         0,
                                         0,
                                         1,
                                         vk::DescriptorType::eUniformBuffer,
                                         nullptr,
                                         &buffer_info,
                                         nullptr};

      device_->updateDescriptorSets(1, &write, 0, nullptr);
    }
  }

  auto create_uniform_buffers() -> void
  {
    vk::DeviceSize buffer_size = sizeof(UniformBufferObject);

    const auto images_count = swapchain_images_.size();
    uniform_buffers_.resize(images_count);
    uniform_buffers_memory_.resize(images_count);

    for (std::size_t i = 0; i < images_count; ++i) {
      std::tie(uniform_buffers_[i], uniform_buffers_memory_[i]) = create_buffer(
          *device_, buffer_size, vk::BufferUsageFlagBits::eUniformBuffer,
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

      vk::ClearValue clear_color{
          vk::ClearColorValue(std::array{0.F, 0.F, 0.F, 1.F})};
      render_pass_begin_info.setClearValueCount(1).setPClearValues(
          &clear_color);

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
      res = platform_.get_resolution();
      glfwWaitEvents();
    }

    device_->waitIdle();

    swapchain_.reset();

    create_swap_chain();
    create_image_views();
    create_render_pass();
    create_graphics_pipeline();
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
        fmt::print(stderr, "Required Validation layer ({}) not found",
                   layerName);
        result = false;
      }
    }

    return result;
  }

  [[nodiscard]] auto create_buffer(const vk::Device& device,
                                   vk::DeviceSize size,
                                   const vk::BufferUsageFlags& usages,
                                   const vk::MemoryPropertyFlags& properties)
      -> std::tuple<vk::UniqueBuffer, vk::UniqueDeviceMemory>;
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

std::tuple<vk::UniqueBuffer, vk::UniqueDeviceMemory>
Application::create_buffer(const vk::Device& device, vk::DeviceSize size,
                           const vk::BufferUsageFlags& usages,
                           const vk::MemoryPropertyFlags& properties)
{
  const vk::BufferCreateInfo create_info{
      {}, size, usages, vk::SharingMode::eExclusive};

  auto buffer = device.createBufferUnique(create_info);

  const auto memory_requirement = device.getBufferMemoryRequirements(*buffer);

  const vk::MemoryAllocateInfo alloc_info{
      memory_requirement.size,
      find_memory_type(memory_requirement.memoryTypeBits, properties)};
  auto buffer_memory = device.allocateMemoryUnique(alloc_info);
  device.bindBufferMemory(*buffer, *buffer_memory, 0);
  return {std::move(buffer), std::move(buffer_memory)};
}
