#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

class DirectionalLightSphere
{
public:
	DirectionalLightSphere(int x, int y, int width, int height, int windowWidth, int windowHeight);

	void GetLightDirection(glm::fvec3& direction, double xpos, double ypos);
	void Draw();

	void SetMatrixView(const glm::fmat4& matrix);
	void SetMatrixProjection(const glm::fmat4& matrix);
	const glm::fmat4& GetMatrixView() const { return m_view; }
	const glm::fmat4& GetMatrixProjection() const { return m_projection; }

private:
	float m_radius;

	int m_viewportWidth;
	int m_viewportHeight;
	int m_viewportX;
	int m_viewportY;
	int m_windowWidth;
	int m_windowHeight;

	GLuint m_VAO;
	GLuint m_IBO;
	GLuint m_verticesVBO;
	GLuint m_normalsVBO;
	int m_numIndex;

	glm::fmat4 m_view;
	glm::fmat4 m_projection;
	glm::fmat4 m_invVP;

private:
	void LoadSphere();

};