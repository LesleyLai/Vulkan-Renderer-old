#ifndef BUFFER_UTILS_HPP
#define BUFFER_UTILS_HPP

#include <vulkan/vulkan.hpp>

namespace vulkan {

[[nodiscard]] auto create_buffer(vk::PhysicalDevice physical_device,
                                 vk::Device device, vk::DeviceSize size,
                                 vk::BufferUsageFlags usages,
                                 vk::MemoryPropertyFlags properties)
    -> std::tuple<vk::UniqueBuffer, vk::UniqueDeviceMemory>;

// Copy a vulkan device buffer to another
// queue is the Vulkan queue to submit to
auto copy_buffer(vk::Device device, vk::Queue queue,
                 vk::CommandPool command_pool, vk::Buffer src, vk::Buffer dst,
                 vk::DeviceSize size) -> void;

[[nodiscard]] auto
create_buffer_from_data(vk::PhysicalDevice physical_device, vk::Device device,
                        vk::Queue queue, vk::CommandPool command_pool,
                        vk::BufferUsageFlags usages, const void* data,
                        vk::DeviceSize size)
    -> std::tuple<vk::UniqueBuffer, vk::UniqueDeviceMemory>;

} // namespace vulkan

#endif // BUFFER_UTILS_HPP
