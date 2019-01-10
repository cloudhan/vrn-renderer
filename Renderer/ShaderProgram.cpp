#include <iostream>
#include <fstream>
#include <sstream>

#include "ShaderProgram.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using std::cout;
using std::endl;

using std::fstream;
using std::stringstream;
using std::string;


Shader::Shader(const char* path, GLenum type)
	: sid(glCreateShader(type))
{
	fstream in;
	stringstream ss;

	in.open(path);
	ss << in.rdbuf();
	in.close();

	shader_src = ss.str();
	const GLchar* vs_src = shader_src.c_str();
	glShaderSource(sid, 1, &vs_src, nullptr);
	glCompileShader(sid);
	GLint success;
	glGetShaderiv(sid, GL_COMPILE_STATUS, &success);
	if (!success) {
		GLchar info[2048];
		glGetShaderInfoLog(sid, 2048, nullptr, info);
		std::cerr << info << std::endl;
	}
}


Shader::~Shader()
{
	glDeleteShader(sid);
}


GLuint Shader::GetId()
{
	return sid;
}


ShaderProgram::ShaderProgram()
	: m_pid(glCreateProgram())
{ }


ShaderProgram::ShaderProgram(const string & vertPath, const string & fragPath)
	: m_pid(glCreateProgram())
{
	AttachSahder(vertPath.c_str(), GL_VERTEX_SHADER);
	AttachSahder(fragPath.c_str(), GL_FRAGMENT_SHADER);
	Link();
}

ShaderProgram::~ShaderProgram()
{
	glDeleteProgram(m_pid);
}

void ShaderProgram::AttachSahder(const char* path, GLenum type)
{
	Shader temp_shader(path, type);
	glAttachShader(m_pid, temp_shader.GetId());
}

int ShaderProgram::Link()
{
	GLint success;
	glLinkProgram(m_pid);
	glGetProgramiv(m_pid, GL_LINK_STATUS, &success);
	if (!success) {
		GLchar info[2048];
		glGetProgramInfoLog(m_pid, 2048, nullptr, info);
		std::cerr << info << std::endl;
		return -1;
	}
	return 0;
}

void ShaderProgram::Use()
{
	glUseProgram(m_pid);
}

GLuint ShaderProgram::GetId()
{
	return m_pid;
}

void ShaderProgram::SetMatrixModel(const glm::fmat4 & matrix)
{
	// FIXME: fix this in shader!
	std::cout << "ERROR: model matrix should always be an identity" << std::endl;
	GLuint u_model = GetUniformLocation("model");
	glUniformMatrix4fv(u_model, 1, GL_FALSE, glm::value_ptr(matrix));
}

void ShaderProgram::SetMatrixView(const glm::fmat4 & matrix)
{
	GLuint u_view = GetUniformLocation("view");
	glUniformMatrix4fv(u_view, 1, GL_FALSE, glm::value_ptr(matrix));

}

void ShaderProgram::SetMatrixProjection(const glm::fmat4 & matrix)
{
	GLuint u_perspective = GetUniformLocation("perspective");
	glUniformMatrix4fv(u_perspective, 1, GL_FALSE, glm::value_ptr(matrix));
}

void ShaderProgram::SetDefaults()
{
	auto model = glm::fmat4x4(1.0);
	auto view = glm::lookAt(glm::fvec3{ 96, 96, 300 }, { 96,96,0 }, { 0, -1, 0 });
	auto perspective = glm::perspective<float>(glm::pi<float>() / 4.0f, 1.0f, 0.01f, 1000.0f);

	SetMatrixModel(model);
	SetMatrixView(view);
	SetMatrixProjection(perspective);
	SetCameraPosition({ 96.0f, 96.0f, 300.0f });
	SetDirectionalLight({ 192.0f, 192.0f, 500.0f });
	SetTextureUnit();
}

void ShaderProgram::SetCameraPosition(const glm::fvec3 & cameraPosition)
{
	auto uid = GetUniformLocation("cameraPosition");
	glUniform3fv(uid, 1, glm::value_ptr(cameraPosition));
}

void ShaderProgram::SetDirectionalLight(const glm::fvec3 & direction)
{
	auto uid = GetUniformLocation("lightDirection");
	glUniform3fv(uid, 1, glm::value_ptr(direction));
}

void ShaderProgram::SetIsTextured(bool flag)
{
	auto uid = GetUniformLocation("isTextured");
	glUniform1i(uid, flag);
}

void ShaderProgram::SetIsSpeculared(bool flag)
{
	auto uid = GetUniformLocation("isSpeculared");
	glUniform1i(uid, flag);
}

void ShaderProgram::SetTextureUnit()
{
	auto uid = GetUniformLocation("textureDiffuse");
	glUniform1i(uid, 0); // set texture unit 0 for diffuse texture
}

GLuint ShaderProgram::GetUniformLocation(const char * uniformName)
{
	GLuint uid = glGetUniformLocation(m_pid, uniformName);
	if (uid == (unsigned int)(-1))
	{
		std::cerr << "Unable to get uniform location \"" << uniformName << "\"" << std::endl;
		// we might raise exception here
		return -1;
	}
	return uid;
}
