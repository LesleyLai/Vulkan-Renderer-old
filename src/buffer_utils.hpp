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

[[nodiscard]] auto
create_image(vk::PhysicalDevice physical_device, vk::Device device,
             std::uint32_t width, std::uint32_t height, vk::Format format,
             vk::ImageTiling tiling, vk::ImageUsageFlags usage,
             vk::MemoryPropertyFlags properties)
    -> std::tuple<vk::UniqueImage, vk::UniqueDeviceMemory>;

[[nodiscard]] auto create_image_view(vk::Device device, vk::Image image,
                                     vk::Format format,
                                     vk::ImageAspectFlags image_aspect)
    -> vk::UniqueImageView;

void copy_buffer_to_image(vk::Device device, vk::Queue queue,
                          vk::CommandPool command_pool, vk::Buffer buffer,
                          vk::Image image, std::uint32_t width,
                          std::uint32_t height);

void transition_image_layout(vk::Device device, vk::Queue queue,
                             vk::CommandPool command_pool, vk::Image image,
                             vk::Format format, vk::ImageLayout old_layout,
                             vk::ImageLayout new_layout);

} // namespace vulkan

#endif // BUFFER_UTILS_HPP
