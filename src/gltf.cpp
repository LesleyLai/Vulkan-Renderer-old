#include "gltf.hpp"

#include <fmt/format.h>

#include <stb_image.h>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>

namespace vulkan {

[[nodiscard]] auto load_gltf_files(std::string_view file_name) -> Model
{
  const auto gltf_model = [file_name = file_name.data()]() {
    tinygltf::Model gltf_model;
    tinygltf::TinyGLTF gltf_loader;
    std::string err;
    std::string warn;

    bool ret =
        gltf_loader.LoadASCIIFromFile(&gltf_model, &err, &warn, file_name);
    if (!ret) {
      throw std::runtime_error{"Tiny GLTF " + err};
    }

    return gltf_model;
  }();

  return Model{};
}

} // namespace vulkan
