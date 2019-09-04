#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <string_view>

[[nodiscard]] auto read_file(std::string_view filename) -> std::string;

#endif // UTILS_HPP
