#pragma once

#include <glad/glad.h>
#include <Eigen/Core>
#include <string>

class FaceModel
{
public:
	FaceModel(const Eigen::MatrixXd& vertices, const Eigen::MatrixXd& normals,
		const Eigen::MatrixXi& indices, const std::string& texture_path);
	~FaceModel();

	void Use();
	void Draw();
	void LoadMesh(const Eigen::MatrixXd& vertices, const Eigen::MatrixXd& normals, const Eigen::MatrixXi& indices);

private:
	GLuint m_VAO;
	GLuint m_IBO;
	GLuint m_verticesVBO;
	GLuint m_normalsVBO;

	GLuint m_textureId;

	int m_numIndex;
};

