#include <fstream>
#include <string>

#include "shader_module.hpp"
#include "utils.hpp"

namespace vulkan {

[[nodiscard]] auto create_shader_module_from_file(std::string_view filename,
                                                  const vk::Device& device)
    -> vk::UniqueShaderModule
{
  return create_shader_module(read_file(filename), device);
}

[[nodiscard]] auto create_shader_module(std::string_view code,
                                        const vk::Device& device)
    -> vk::UniqueShaderModule
{
  vk::ShaderModuleCreateInfo create_info{
      {}, code.size(), reinterpret_cast<const std::uint32_t*>(code.data())};

  return device.createShaderModuleUnique(create_info);
}

} // namespace vulkan
