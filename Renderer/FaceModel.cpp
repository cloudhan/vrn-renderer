#include "FaceModel.h"
#include <vector>

#include <igl/png/texture_from_png.h>

using namespace Eigen;
using namespace std;


FaceModel::FaceModel(const MatrixXd& vertices, const MatrixXd& normals, 
	const MatrixXi& indices, const string& texture_path)
{
	glGenVertexArrays(1, &m_VAO);
	glGenBuffers(1, &m_IBO);
	glGenBuffers(1, &m_verticesVBO);
	glGenBuffers(1, &m_normalsVBO);

	LoadMesh(vertices, normals, indices);

	igl::png::texture_from_png(texture_path, m_textureId);
}

FaceModel::~FaceModel()
{
}

void FaceModel::Use()
{
	glBindVertexArray(m_VAO);
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IBO);
}

void FaceModel::Draw()
{
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_textureId);

	glBindVertexArray(m_VAO);
	glDrawElements(GL_TRIANGLES, m_numIndex, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}

void FaceModel::LoadMesh(const MatrixXd & vertices, const MatrixXd& normals, const MatrixXi & indices)
{
	MatrixXf vf;
	if (vertices.Options & Eigen::RowMajor) { vf = vertices.template cast<float>(); }
	else { vf = vertices.transpose().template cast<float>(); }

	MatrixXf nf;
	// we need flip all normal manually;
	if (normals.Options & Eigen::RowMajor) { nf = -normals.template cast<float>(); }
	else { nf = -normals.transpose().template cast<float>(); }

	m_numIndex = indices.size();
	MatrixXi idx;
	if (indices.Options & Eigen::RowMajor) { idx = indices; }
	else { idx = indices.transpose(); }

	glBindVertexArray(m_VAO);
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_verticesVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vf.size(), vf.data(), GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 3, (GLvoid*)(sizeof(GLfloat) * 0));
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, m_normalsVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * nf.size(), nf.data(), GL_STATIC_DRAW);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 3, (GLvoid*)(sizeof(GLfloat) * 0));
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int) * idx.size(), idx.data(), GL_STATIC_DRAW);
	}
	glBindVertexArray(0);
}
