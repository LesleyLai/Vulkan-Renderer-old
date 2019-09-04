#include "gltf.hpp"

#include <iostream>

#include <fstream>
#include <rapidjson/document.h>

#include "utils.hpp"

[[nodiscard]] auto load_gltf_scene(std::string_view filename) -> GltfScene
{
  rapidjson::Document document;

  document.Parse(read_file(filename).data());
  std::cout << document["asset"]["version"].GetString() << std::endl;
  return {};
}
