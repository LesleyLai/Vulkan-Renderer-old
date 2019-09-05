#include "utils.hpp"

#include <fstream>

[[nodiscard]] auto read_file(std::string_view file_location) -> std::string
{
  std::ifstream file(file_location.data(), std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file: " +
                             std::string{file_location});
  }

  size_t file_size = static_cast<size_t>(file.tellg());
  std::string buffer;
  buffer.resize(file_size);

  file.seekg(0);
  file.read(buffer.data(), static_cast<std::streamsize>(file_size));

  return buffer;
}
