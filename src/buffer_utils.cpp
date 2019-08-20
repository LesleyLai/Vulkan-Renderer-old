#include "buffer_utils.hpp"

namespace vulkan {

template <typename Func>
auto submit_one_time_commands(vk::Device device, vk::Queue queue,
                              vk::CommandPool command_pool, Func&& f) -> void
{
  const vk::CommandBufferAllocateInfo alloc_info{
      command_pool, vk::CommandBufferLevel::ePrimary, 1};

  vk::CommandBuffer command_buffer;
  device.allocateCommandBuffers(&alloc_info, &command_buffer);

  const vk::CommandBufferBeginInfo begin_info{
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
  command_buffer.begin(begin_info);

  std::forward<Func>(f)(command_buffer);

  command_buffer.end();

  vk::SubmitInfo submit_info;
  submit_info.setCommandBufferCount(1).setPCommandBuffers(&command_buffer);
  queue.submit(1, &submit_info, vk::Fence{});
  queue.waitIdle();

  device.freeCommandBuffers(command_pool, 1, &command_buffer);
}

static auto find_memory_type(vk::PhysicalDevice physical_device,
                             uint32_t type_filter,
                             vk::MemoryPropertyFlags properties)
    -> std::uint32_t
{
  const auto device_memory_properties = physical_device.getMemoryProperties();

  for (uint32_t i = 0; i < device_memory_properties.memoryTypeCount; i++) {
    if (type_filter & (1 << i) &&
        (device_memory_properties.memoryTypes[i].propertyFlags & properties) ==
            properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}

[[nodiscard]] auto create_buffer(vk::PhysicalDevice physical_device,
                                 vk::Device device, vk::DeviceSize size,
                                 vk::BufferUsageFlags usages,
                                 vk::MemoryPropertyFlags properties)
    -> std::tuple<vk::UniqueBuffer, vk::UniqueDeviceMemory>
{
  const vk::BufferCreateInfo create_info{
      {}, size, usages, vk::SharingMode::eExclusive};

  auto buffer = device.createBufferUnique(create_info);

  const auto memory_requirement = device.getBufferMemoryRequirements(*buffer);

  const vk::MemoryAllocateInfo alloc_info{
      memory_requirement.size,
      find_memory_type(physical_device, memory_requirement.memoryTypeBits,
                       properties)};
  auto buffer_memory = device.allocateMemoryUnique(alloc_info);
  device.bindBufferMemory(*buffer, *buffer_memory, 0);
  return {std::move(buffer), std::move(buffer_memory)};
}

auto copy_buffer(vk::Device device, vk::Queue queue,
                 vk::CommandPool command_pool, vk::Buffer src, vk::Buffer dst,
                 vk::DeviceSize size) -> void
{
  const vk::CommandBufferAllocateInfo alloc_info{
      command_pool, vk::CommandBufferLevel::ePrimary, 1};

  const auto command_buffers = device.allocateCommandBuffersUnique(alloc_info);
  const auto& command_buffer = command_buffers[0].get();

  const vk::CommandBufferBeginInfo begin_info{
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
  command_buffer.begin(begin_info);
  vk::BufferCopy copy_region{0, 0, size};
  command_buffer.copyBuffer(src, dst, 1, &copy_region);
  command_buffer.end();

  vk::SubmitInfo submit_info;
  submit_info.setCommandBufferCount(1).setPCommandBuffers(&command_buffer);
  queue.submit(1, &submit_info, vk::Fence{});
  queue.waitIdle();
}

[[nodiscard]] auto create_buffer_from_data(
    vk::PhysicalDevice physical_device, vk::Device device, vk::Queue queue,
    vk::CommandPool command_pool, vk::BufferUsageFlags usages, const void* data,
    vk::DeviceSize size) -> std::tuple<vk::UniqueBuffer, vk::UniqueDeviceMemory>
{
  const auto [staging_buffer, staging_buffer_memory] = vulkan::create_buffer(
      physical_device, device, size, vk::BufferUsageFlagBits::eTransferSrc,
      vk::MemoryPropertyFlagBits::eHostVisible |
          vk::MemoryPropertyFlagBits::eHostCoherent);

  void* mapped_data;
  device.mapMemory(*staging_buffer_memory, 0, size, {}, &mapped_data);
  memcpy(mapped_data, data, size);
  device.unmapMemory(*staging_buffer_memory);

  auto [buffer, buffer_memory] =
      vulkan::create_buffer(physical_device, device, size,
                            usages | vk::BufferUsageFlagBits::eTransferDst,
                            vk::MemoryPropertyFlagBits::eDeviceLocal);

  vulkan::copy_buffer(device, queue, command_pool, *staging_buffer, *buffer,
                      size);
  return {std::move(buffer), std::move(buffer_memory)};
}

[[nodiscard]] auto
create_image(vk::PhysicalDevice physical_device, vk::Device device,
             std::uint32_t width, std::uint32_t height, vk::Format format,
             vk::ImageTiling tiling, vk::ImageUsageFlags usage,
             vk::MemoryPropertyFlags properties)
    -> std::tuple<vk::UniqueImage, vk::UniqueDeviceMemory>
{
  const vk::ImageCreateInfo image_create_info{{},
                                              vk::ImageType::e2D,
                                              format,
                                              vk::Extent3D{width, height, 1},
                                              1,
                                              1,
                                              vk::SampleCountFlagBits::e1,
                                              tiling,
                                              usage,
                                              vk::SharingMode::eExclusive,
                                              0,
                                              nullptr,
                                              vk::ImageLayout::eUndefined};
  auto image = device.createImageUnique(image_create_info);
  const auto memory_requirement = device.getImageMemoryRequirements(*image);

  const vk::MemoryAllocateInfo malloc_info{
      memory_requirement.size,
      find_memory_type(physical_device, memory_requirement.memoryTypeBits,
                       properties)};
  auto image_memory = device.allocateMemoryUnique(malloc_info);
  device.bindImageMemory(*image, *image_memory, 0);

  return {std::move(image), std::move(image_memory)};
}

[[nodiscard]] auto create_image_view(vk::Device device, vk::Image image,
                                     vk::Format format,
                                     vk::ImageAspectFlags image_aspect)
    -> vk::UniqueImageView
{
  vk::ImageSubresourceRange subresource_range;
  subresource_range.setAspectMask(image_aspect)
      .setBaseMipLevel(0)
      .setLevelCount(1)
      .setBaseArrayLayer(0)
      .setLayerCount(1);

  vk::ImageViewCreateInfo create_info;
  create_info.setImage(image)
      .setViewType(vk::ImageViewType::e2D)
      .setFormat(format)
      .setComponents(vk::ComponentMapping{})
      .setSubresourceRange(subresource_range);

  return device.createImageViewUnique(create_info);
}

void copy_buffer_to_image(vk::Device device, vk::Queue queue,
                          vk::CommandPool command_pool, vk::Buffer buffer,
                          vk::Image image, std::uint32_t width,
                          std::uint32_t height)
{
  submit_one_time_commands(
      device, queue, command_pool,
      [&buffer, &image, width,
       height](const vk::CommandBuffer& command_buffer) {
        vk::BufferImageCopy region;
        region.setBufferOffset(0)
            .setBufferRowLength(0)
            .setBufferImageHeight(0)
            .setImageSubresource(vk::ImageSubresourceLayers{
                vk::ImageAspectFlagBits::eColor, 0, 0, 1})
            .setImageOffset(vk::Offset3D{0, 0, 0})
            .setImageExtent(vk::Extent3D{width, height, 1});

        command_buffer.copyBufferToImage(
            buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);
      });
}

void transition_image_layout(vk::Device device, vk::Queue queue,
                             vk::CommandPool command_pool, vk::Image image,
                             vk::Format format, vk::ImageLayout old_layout,
                             vk::ImageLayout new_layout)
{
  submit_one_time_commands(
      device, queue, command_pool,
      [&](const vk::CommandBuffer& command_buffer) {
        vk::ImageMemoryBarrier barrier;

        barrier.setOldLayout(old_layout)
            .setNewLayout(new_layout)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(image)
            .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

        if (new_layout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
          barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
        } else {
          barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        }

        vk::PipelineStageFlags source_stage;
        vk::PipelineStageFlags destination_stage;

        if (old_layout == vk::ImageLayout::eUndefined &&
            new_layout == vk::ImageLayout::eTransferDstOptimal) {
          barrier.setSrcAccessMask({}).setDstAccessMask(
              vk::AccessFlagBits::eTransferWrite);

          source_stage = vk::PipelineStageFlagBits::eTopOfPipe;
          destination_stage = vk::PipelineStageFlagBits::eTransfer;
        } else if (old_layout == vk::ImageLayout::eUndefined &&
                   new_layout ==
                       vk::ImageLayout::eDepthStencilAttachmentOptimal) {
          barrier.setSrcAccessMask({}).setDstAccessMask(
              vk::AccessFlagBits::eDepthStencilAttachmentRead |
              vk::AccessFlagBits::eDepthStencilAttachmentWrite);

          source_stage = vk::PipelineStageFlagBits::eTopOfPipe;
          destination_stage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
        } else if (old_layout == vk::ImageLayout::eTransferDstOptimal &&
                   new_layout == vk::ImageLayout::eShaderReadOnlyOptimal) {

          barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
              .setDstAccessMask(vk::AccessFlagBits::eShaderRead);

          source_stage = vk::PipelineStageFlagBits::eTransfer;
          destination_stage = vk::PipelineStageFlagBits::eFragmentShader;
        } else {
          throw std::invalid_argument("unsupported layout transition!");
        }

        command_buffer.pipelineBarrier(source_stage, destination_stage, {}, 0,
                                       nullptr, 0, nullptr, 1, &barrier);
      });
}

} // namespace vulkan
