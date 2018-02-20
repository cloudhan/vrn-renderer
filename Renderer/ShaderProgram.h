#pragma once

#include <string>
#include <glad/glad.h>
#include <glm/glm.hpp>

class Shader
{
public:
	Shader(const char* path, GLenum type);
	~Shader();

	GLuint GetId();

private:
	GLuint sid;
	std::string shader_src;
};


class ShaderProgram
{
public:
	ShaderProgram();
	ShaderProgram(const ShaderProgram& sp) = delete;
	ShaderProgram(const std::string& vertPath, const std::string& fragPath);
	~ShaderProgram();

	void AttachSahder(const char * path, GLenum type);
	int Link();
	void Use();
	GLuint GetId();

	void SetDefaults();
	void SetMatrixModel(const glm::fmat4& matrix);
	void SetMatrixView(const glm::fmat4& matrix);
	void SetMatrixProjection(const glm::fmat4& matrix);
	void SetCameraPosition(const glm::fvec3& cameraPosition);
	void SetDirectionalLight(const glm::fvec3& direction);
	void SetIsTextured(bool flag);
	void SetTextureUnit();

private:
	GLuint m_pid;


	GLuint GetUniformLocation(const char* uniformName);
};