#ifndef SHADER_MODULE_HPP
#define SHADER_MODULE_HPP

#include <string>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace vulkan {

[[nodiscard]] auto create_shader_module_from_file(std::string_view filename,
                                                  const vk::Device& device)
    -> vk::UniqueShaderModule;

[[nodiscard]] auto create_shader_module(std::string_view code,
                                        const vk::Device& device)
    -> vk::UniqueShaderModule;

} // namespace vulkan

#endif // SHADER_MODULE_HPP
