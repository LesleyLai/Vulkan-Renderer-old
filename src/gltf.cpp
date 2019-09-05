#include "gltf.hpp"

#include <rapidjson/document.h>

#include <filesystem>
#include <tuple>
#include <vector>

#include "utils.hpp"
#include <fmt/format.h>

namespace fs = std::filesystem;

[[nodiscard]] auto load_gltf_scene(std::string_view file_location) -> GltfScene
{
  rapidjson::Document document;

  fs::path path{file_location};
  const std::string file_name = path.filename().string();
  const std::string file_path = path.remove_filename().string();

  document.Parse(read_file(file_location).data());

  const auto& buffers = document["buffers"].GetArray();
  for (const auto& buffer : buffers) {
    std::vector<std::byte> data;
    data.reserve(static_cast<std::size_t>(buffer["byteLength"].GetInt()));
    std::string_view uri{buffer["uri"].GetString()};

    constexpr std::string_view data_buffer_prefix{"data"};
    if (uri.starts_with(data_buffer_prefix)) {
      throw std::runtime_error{"Data Uri is currently not supported"};
    }
  }

  return {};
}
