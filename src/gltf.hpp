#ifndef GLTF_HPP
#define GLTF_HPP

#include <string_view>

struct GltfScene {
};

[[nodiscard]] auto load_gltf_scene(std::string_view filename) -> GltfScene;

#endif // GLTF_HPP
