#include "graphics_pipeline.hpp"
#include "shader_module.hpp"

namespace vulkan {

[[nodiscard]] auto create_graphics_pipeline_layout(
    vk::Device device, vk::DescriptorSetLayout descriptor_set_layout) noexcept
    -> vk::UniquePipelineLayout
{
  vk::PipelineLayoutCreateInfo pipeline_layout_create_info;
  pipeline_layout_create_info.setSetLayoutCount(1)
      .setPSetLayouts(&descriptor_set_layout)
      .setPushConstantRangeCount(0);

  return device.createPipelineLayoutUnique(pipeline_layout_create_info);
}

[[nodiscard]] auto create_graphics_pipeline(
    vk::Device device, vk::RenderPass render_pass,
    vk::PrimitiveTopology primitive_topology,
    vk::PipelineLayout pipeline_layout, vk::Viewport viewport,
    vk::Rect2D scissor, const GraphicsPipelineShaders& shaders,
    const VertexInputInfo& vertex_input_info) -> vk::UniquePipeline
{
  const auto binding_description = vertex_input_info.binding_description;
  const auto attribute_descriptions = vertex_input_info.attribute_descriptions;

  const vk::PipelineVertexInputStateCreateInfo vertex_input_stage_create_info{
      {},
      1,
      &binding_description,
      static_cast<uint32_t>(attribute_descriptions.size()),
      attribute_descriptions.data()};

  const vk::PipelineInputAssemblyStateCreateInfo input_assembly{
      {}, primitive_topology, false};

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
      .setFrontFace(vk::FrontFace::eCounterClockwise)
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

  vk::PipelineDepthStencilStateCreateInfo depth_stencil_info;
  depth_stencil_info.setDepthTestEnable(true)
      .setDepthWriteEnable(true)
      .setDepthCompareOp(vk::CompareOp::eLess)
      .setDepthBoundsTestEnable(false)
      .setStencilTestEnable(false);

  std::vector<vk::PipelineShaderStageCreateInfo> shader_stages;

  vk::PipelineShaderStageCreateInfo vert_shader_stage_info;
  vert_shader_stage_info.setStage(vk::ShaderStageFlagBits::eVertex)
      .setModule(shaders.vertex)
      .setPName("main");
  shader_stages.push_back(vert_shader_stage_info);

  vk::PipelineShaderStageCreateInfo frag_shader_stage_info;
  frag_shader_stage_info.setStage(vk::ShaderStageFlagBits::eFragment)
      .setModule(shaders.fragment)
      .setPName("main");
  shader_stages.push_back(frag_shader_stage_info);

  if (shaders.tess) {
    throw std::runtime_error{
        "Support for tessellation shaders is not implemented yet!"};
  }

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
      .setPDepthStencilState(&depth_stencil_info)
      .setLayout(pipeline_layout)
      .setRenderPass(render_pass)
      .setSubpass(0)
      .setBasePipelineHandle(nullptr);

  return device.createGraphicsPipelineUnique(nullptr, pipeline_create_info);
}

} // namespace vulkan
