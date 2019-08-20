#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/vec3.hpp>

#include <cmath>
#include <vector>

#include <fmt/format.h>

// Default camera values
constexpr float yaw = -90.0f;
constexpr float pitch = 0.0f;
constexpr float speed = 2.5f;
constexpr float sensitivity = 0.1f;
constexpr float zoom = 45.0f;

class Camera {
public:
  enum class Movement { forward, backward, left, right };

  // Constructor with vectors
  explicit Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f),
                  glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = ::yaw,
                  float pitch = ::pitch)
      : position_{position}, front_{glm::vec3(0.0f, 0.0f, -1.0f)}, right_{},
        world_up_{up}, yaw_{yaw}, pitch_{pitch}, speed_{10.f},
        mouse_sensitivity_{sensitivity}, zoom_{::zoom}
  {
    update_camera_vectors();
  }

  // Returns the view matrix calculated using Euler Angles and the LookAt Matrix
  [[nodiscard]] auto view_matrix() -> glm::mat4
  {
    return glm::lookAt(position_, position_ + front_, up_);
  }

  // Processes input received from any keyboard-like input system. Accepts input
  // parameter in the form of camera defined ENUM (to abstract it from windowing
  // systems)
  auto move(Movement direction, float delta_time) -> void
  {
    float dx = speed_ * delta_time;
    switch (direction) {
    case Camera::Movement::forward:
      position_ += front_ * dx;
      break;
    case Camera::Movement::backward:
      position_ -= front_ * dx;
      break;
    case Camera::Movement::left:
      position_ -= right_ * dx;
      break;
    case Camera::Movement::right:
      position_ += right_ * dx;
      break;
    }
  }

  // Processes input received from a mouse input system. Expects the offset
  // value in both the x and y direction.
  auto mouse_movement(float xoffset, float yoffset, bool constrain_pitch = true)
      -> void
  {
    xoffset *= mouse_sensitivity_;
    yoffset *= mouse_sensitivity_;

    yaw_ += xoffset;
    pitch_ += yoffset;

    // Make sure that when pitch is out of bounds, screen doesn't get flipped
    if (constrain_pitch) {
      if (pitch_ > 89.0f) {
        pitch_ = 89.0f;
      }
      if (pitch_ < -89.0f) {
        pitch_ = -89.0f;
      }
    }

    // Update Front, Right and Up Vectors using the updated Euler angles
    update_camera_vectors();
  }

  // Processes input received from a mouse scroll-wheel event. Only requires
  // input on the vertical wheel-axis
  void mouse_scroll(float yoffset)
  {
    if (zoom_ >= 1.0f && zoom_ <= 45.0f) {
      zoom_ -= yoffset;
    }
    if (zoom_ <= 1.0f) {
      zoom_ = 1.0f;
    }
    if (zoom_ >= 45.0f) {
      zoom_ = 45.0f;
    }
  }

  float zoom()
  {
    return zoom_;
  }

private:
  // Camera Attributes
  glm::vec3 position_;
  glm::vec3 front_;
  glm::vec3 up_;
  glm::vec3 right_;
  glm::vec3 world_up_;
  // Euler Angles
  float yaw_;
  float pitch_;
  // Camera options
  float speed_;
  float mouse_sensitivity_;
  float zoom_;

  // Calculates the front vector from the Camera's (updated) Euler Angles
  void update_camera_vectors()
  {
    // Calculate the new Front vector
    glm::vec3 front;
    front.x = std::cos(glm::radians(yaw_)) * std::cos(glm::radians(pitch_));
    front.y = std::sin(glm::radians(pitch_));
    front.z = std::sin(glm::radians(yaw_)) * std::cos(glm::radians(pitch_));
    front_ = glm::normalize(front);
    // Also re-calculate the Right and Up vector
    right_ = glm::normalize(glm::cross(
        front_, world_up_)); // Normalize the vectors, because their length gets
                             // closer to 0 the more you look up or down which
                             // results in slower movement.
    up_ = glm::normalize(glm::cross(right_, front_));
  }
};
#endif // CAMERA_HPP
