#define _USE_MATH_DEFINES
#include <cmath>

#include "DirectionalLightSphere.h"

#include <memory>
#include <vector>
#include <iostream>

namespace {
float M_PIDIV2 = M_PI / 2.0;
float M_2PI = M_PI * 2.0;
}

DirectionalLightSphere::DirectionalLightSphere(int x, int y, int width, int height, int windowWidth, int windowHeight)
	: m_radius(1.0)
	, m_viewportWidth(width)
	, m_viewportHeight(height)
	, m_windowWidth(windowWidth)
	, m_windowHeight(windowHeight)
	, m_viewportX(x)
	, m_viewportY(y)
	, m_view(glm::fmat4(1.0))
	, m_projection(glm::fmat4(1.0))
	, m_invVP(glm::fmat4(1.0))
{
	glGenVertexArrays(1, &m_VAO);
	glGenBuffers(1, &m_IBO);
	glGenBuffers(1, &m_verticesVBO);
	glGenBuffers(1, &m_normalsVBO);

	LoadSphere();
}

void DirectionalLightSphere::GetLightDirection(glm::fvec3 & direction, double xpos, double ypos)
{
	ypos = m_windowHeight - ypos;


	if (xpos >= m_viewportX && xpos < m_viewportX + m_viewportWidth &&
		ypos >= m_viewportY && ypos < m_viewportY + m_viewportHeight)
	{
		double ndcX, ndcY;
		ndcX = (xpos - m_viewportX) / m_viewportWidth * 2.0 - 1.0;
		ndcY = (ypos - m_viewportY) / m_viewportHeight * 2.0 - 1.0;
		glm::fvec4 ndcCoord = { ndcX, ndcY, 0.0, 1.0 };
		auto worldCoord = m_invVP * ndcCoord;

		float temp = m_radius * m_radius - worldCoord.x * worldCoord.x - worldCoord.y * worldCoord.y;
		direction.x = worldCoord.x;
		direction.y = worldCoord.y;
		if (temp > 1e-5)
			direction.z = sqrt(temp);
		else
			direction.z = 0;

#ifndef NDEBUG
		std::cout << direction.x << "," << direction.y << "," << direction.z << std::endl;
#endif
	}
}

void DirectionalLightSphere::Draw()
{
	glBindVertexArray(m_VAO);
	glDrawElements(GL_TRIANGLES, m_numIndex, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}

void DirectionalLightSphere::SetMatrixView(const glm::fmat4 & matrix)
{
	m_view = matrix;
	m_invVP = glm::inverse(m_projection * m_view);
}

void DirectionalLightSphere::SetMatrixProjection(const glm::fmat4 & matrix)
{
	m_projection = matrix;
	m_invVP = glm::inverse(m_projection * m_view);
}


void DirectionalLightSphere::LoadSphere()
{
	std::vector<glm::fvec3> vertices;
	std::vector<glm::fvec3> normals;
	std::vector<glm::uvec3> indices;

	const int tessellation = 30;


	size_t verticalSegments = tessellation;
	size_t horizontalSegments = tessellation * 2;

	// Create rings of vertices at progressively higher latitudes.
	for (size_t i = 0; i <= verticalSegments; i++)
	{
		float v = 1 - (float)i / verticalSegments;  //texcoord.y

		float latitude = (i * M_PI / verticalSegments) - M_PIDIV2;
		float dy = glm::sin(latitude);
		float dxz = glm::cos(latitude);

		// Create a single ring of vertices at this latitude.
		for (size_t j = 0; j <= horizontalSegments; j++)
		{
			float u = (float)j / horizontalSegments; // texcoord.x

			float longitude = j * M_2PI / horizontalSegments;
			float dx = glm::sin(longitude);
			float dz = glm::cos(longitude);

			dx *= dxz;
			dz *= dxz;

			glm::fvec3 normal = { dx, dy, dz };
			glm::fvec3 vertex = normal * m_radius;

			vertices.push_back(vertex);
			normals.push_back(normal);
		}
	}

	// Fill the index buffer with triangles joining each pair of latitude rings.
	size_t stride = horizontalSegments + 1;

	for (size_t i = 0; i < verticalSegments; i++)
	{
		for (size_t j = 0; j <= horizontalSegments; j++)
		{
			size_t nextI = i + 1;
			size_t nextJ = (j + 1) % stride;
			
			indices.push_back({ i*stride + j, nextI*stride + j, i*stride + nextJ });
			indices.push_back({ i*stride + nextJ, nextI*stride + j, nextI*stride + nextJ });
		}
	}
	m_numIndex = indices.size() * 3;


	glBindVertexArray(m_VAO);
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_verticesVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::fvec3) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::fvec3), (GLvoid*)(sizeof(GLfloat) * 0));
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, m_normalsVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::fvec3) * normals.size(), normals.data(), GL_STATIC_DRAW);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::fvec3), (GLvoid*)(sizeof(GLfloat) * 0));
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(glm::uvec3) * indices.size(), indices.data(), GL_STATIC_DRAW);
	}
	glBindVertexArray(0);

}
