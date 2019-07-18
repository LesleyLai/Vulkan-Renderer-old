#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>

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
#include "shader_module.hpp"

constexpr std::array validation_layers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool vk_enable_validation_layers = false;
#else
constexpr bool vk_enable_validation_layers = true;
#endif

constexpr std::array device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

constexpr std::size_t frames_in_flight = 2;

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
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
  return VK_FALSE;
}

static void framebuffer_resize_callback(GLFWwindow* window, int width,
                                        int height);

class Application {
public:
  bool frame_buffer_resized = false;

  Application() : instance_{create_instance()}, dldy_{create_dynamic_loader()}
  {
    glfwSetFramebufferSizeCallback(platform_.window(),
                                   framebuffer_resize_callback);
    glfwSetWindowUserPointer(platform_.window(), this);

    setup_debug_messenger();
    surface_ = platform_.create_vulkan_surface(instance_.get(), dldy_);
    physical_device_ = pick_physical_device();
    create_logical_device();
    create_swap_chain();
    create_image_views();
    create_render_pass();
    create_graphics_pipeline();
    create_frame_buffers();
    create_command_pool();
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

  vk::Queue graphics_queue_;
  vk::Queue present_queue_;
  vk::UniqueSwapchainKHR swapchain_;
  std::vector<vk::Image> swapchain_images_;
  vk::Format swapchain_image_format_;
  vk::Extent2D swapchain_extent_;
  std::vector<vk::UniqueImageView> swapchain_image_views_;

  vk::UniqueRenderPass render_pass_;
  vk::UniquePipelineLayout pipeline_layout_;
  vk::UniquePipeline graphics_pipeline_;

  std::vector<vk::UniqueFramebuffer> swapchain_framebuffers_;

  vk::UniqueCommandPool command_pool_;
  std::vector<vk::CommandBuffer> command_buffers_;

  std::array<vk::UniqueSemaphore, 2> image_available_semaphores;
  std::array<vk::UniqueSemaphore, 2> render_finished_semaphores;
  std::array<vk::UniqueFence, 2> in_flight_fences;
  size_t current_frame = 0;

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

  [[nodiscard]] auto pick_physical_device() -> vk::PhysicalDevice
  {
    const auto devices = instance_->enumeratePhysicalDevices();
    assert(!devices.empty());

    for (const auto& device : devices) {
      if (is_physical_device_suitable(device)) {
        return device;
      }
    }

    std::cerr << "Cannot find suitable physical device for Vulkan\n";
    return nullptr;
  }

  void create_logical_device()
  {
    QueueFamilyIndices indices = find_queue_families(physical_device_);

    std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
    std::set<uint32_t> unique_queue_families = {indices.graphics_family.value(),
                                                indices.present_family.value()};

    float queue_priority = 1.0F;
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

  void create_image_views()
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

  void create_render_pass()
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

  void create_graphics_pipeline()
  {
    auto vert_shader_module =
        create_shader_module("shaders/vert.spv", *device_);
    auto frag_shader_module =
        create_shader_module("shaders/frag.spv", *device_);

    vk::PipelineShaderStageCreateInfo vert_shader_stage_info;
    vert_shader_stage_info.setStage(vk::ShaderStageFlagBits::eVertex)
        .setModule(*vert_shader_module)
        .setPName("main");
    vk::PipelineShaderStageCreateInfo frag_shader_stage_info;
    frag_shader_stage_info.setStage(vk::ShaderStageFlagBits::eFragment)
        .setModule(*frag_shader_module)
        .setPName("main");

    const vk::PipelineVertexInputStateCreateInfo vertex_input_stage_create_info{
        {}, 0, nullptr, 0, nullptr};

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
    pipeline_layout_create_info.setSetLayoutCount(0).setPushConstantRangeCount(
        0);

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

  void create_frame_buffers()
  {
    swapchain_framebuffers_.clear();
    swapchain_framebuffers_.reserve(swapchain_image_views_.size());
    for (const auto& image_view : swapchain_image_views_) {
      std::array attachments{*image_view};

      vk::FramebufferCreateInfo create_info;
      create_info.setRenderPass(*render_pass_)
          .setAttachmentCount(attachments.size())
          .setPAttachments(attachments.data())
          .setWidth(swapchain_extent_.width)
          .setHeight(swapchain_extent_.height)
          .setLayers(1);

      swapchain_framebuffers_.emplace_back(
          device_->createFramebufferUnique(create_info));
    }
  }

  void create_command_pool()
  {
    const QueueFamilyIndices queue_family_indices =
        find_queue_families(physical_device_);

    const vk::CommandPoolCreateInfo create_info{
        {}, queue_family_indices.graphics_family.value()};

    command_pool_ = device_->createCommandPoolUnique(create_info);
  }

  void create_command_buffers()
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
          vk::ClearColorValue(std::array{0.f, 0.f, 1.f, 1.f})};
      render_pass_begin_info.setClearValueCount(1).setPClearValues(
          &clear_color);

      command_buffer.beginRenderPass(&render_pass_begin_info,
                                     vk::SubpassContents::eInline);

      command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                  *graphics_pipeline_);

      command_buffer.draw(3, 1, 0, 0);

      command_buffer.endRenderPass();

      command_buffer.end();
    }
  }

  void create_sync_objects()
  {
    vk::SemaphoreCreateInfo semaphore_create_info;
    vk::FenceCreateInfo fence_create_info{vk::FenceCreateFlagBits::eSignaled};
    for (size_t i = 0; i < frames_in_flight; ++i) {
      image_available_semaphores[i] =
          device_->createSemaphoreUnique(semaphore_create_info);
      render_finished_semaphores[i] =
          device_->createSemaphoreUnique(semaphore_create_info);
      in_flight_fences[i] = device_->createFenceUnique(fence_create_info);
    }
  }

  void recreate_swapchain()
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
    create_command_buffers();
  }

  void render()
  {
    device_->waitForFences(1, &(*in_flight_fences[current_frame]), true,
                           std::numeric_limits<uint64_t>::max());

    auto [result, image_index] = device_->acquireNextImageKHR(
        *swapchain_, std::numeric_limits<uint64_t>::max(),
        *image_available_semaphores[current_frame], nullptr);

    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR) {
      recreate_swapchain();
      return;
    }
    assert(result == vk::Result::eSuccess);

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

    result = present_queue_.presentKHR(&present_info);
    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR || frame_buffer_resized) {
      recreate_swapchain();
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
        std::cerr << "Require Validation layer (" << layerName << ") no found";
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
  std::cerr << "Error: " << e.what() << '\n';
} catch (...) {
  std::cerr << "Unknown exception thrown!\n";
}
