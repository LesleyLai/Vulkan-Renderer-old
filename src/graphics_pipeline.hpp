#ifndef GRAPHICS_PIPELINE_HPP
#define GRAPHICS_PIPELINE_HPP

#include <optional>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace vulkan {

struct VertexInputInfo {
  vk::VertexInputBindingDescription binding_description;
  std::vector<vk::VertexInputAttributeDescription> attribute_descriptions;
};

struct TessShaders {
  vk::ShaderModule control;
  vk::ShaderModule eval;
};

struct GraphicsPipelineShaders {
  vk::ShaderModule vertex;
  vk::ShaderModule fragment;
  std::optional<TessShaders> tess;
};

[[nodiscard]] auto create_graphics_pipeline_layout(
    vk::Device device, vk::DescriptorSetLayout descriptor_set_layout) noexcept
    -> vk::UniquePipelineLayout;

[[nodiscard]] auto create_graphics_pipeline(
    vk::Device device, vk::RenderPass render_pass,
    vk::PrimitiveTopology primitive_topology,
    vk::PipelineLayout pipeline_layout, vk::Viewport viewport,
    vk::Rect2D scissor, const GraphicsPipelineShaders& shaders,
    const VertexInputInfo& vertex_input_info) -> vk::UniquePipeline;

} // namespace vulkan

#endif // GRAPHICS_PIPELINE_HPP
