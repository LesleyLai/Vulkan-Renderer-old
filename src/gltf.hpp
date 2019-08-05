#ifndef GLTF_HPP
#define GLTF_HPP

#include <string_view>

namespace vulkan {

struct Model {
};

[[nodiscard]] auto load_gltf_files(std::string_view file_name) -> Model;

}; // namespace vulkan

#endif // GLTF_HPP
