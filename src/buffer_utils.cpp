#include "buffer_utils.hpp"

namespace vulkan {

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

} // namespace vulkan
