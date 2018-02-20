#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#define _USE_MATH_DEFINES
#include <cmath>

#include <iostream>
#include <vector>
#include <memory>

#include <glm/gtc/matrix_transform.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <igl/readOBJ.h>
#include <igl/per_vertex_normals.h>


#include "FaceModel.h"
#include "DirectionalLightSphere.h"
#include "ShaderProgram.h"
#include "Utilities.h"

using namespace Eigen;
using namespace std;

MatrixXd V; // original vertices
MatrixXd U; // updated vertices
MatrixXd N; // per-vertice normals
MatrixXi F; // face indices
SparseMatrix<double> L; // Laplace-Beltrami operator 
SparseMatrix<double> K; 

bool g_isLeftButtonPressed = false;
const int   g_windowWidth  = 480*2;
const int   g_windowHeight = 480;
GLFWwindow* g_pWindow;

glm::fvec3 g_lightDirection = { 1,1,1 };
std::unique_ptr<FaceModel> g_pFaceModel;
std::unique_ptr<ShaderProgram> g_pShaderProgram;
std::unique_ptr<DirectionalLightSphere> g_pDLSphere;

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mode)
{

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, true);
	}

	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
	{
		Utilities::Laplacian::Smooth(U, F, L);
		igl::per_vertex_normals(U, F, N);
		g_pFaceModel->LoadMesh(U, N, F);
	}

	if (key == GLFW_KEY_R && action == GLFW_PRESS)
	{
		U = V;
		igl::per_vertex_normals(V, F, N);
		g_pFaceModel->LoadMesh(V, N, F);
	}

}

static void mouseBtnCallback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
		g_isLeftButtonPressed = true;
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
		g_isLeftButtonPressed = false;
}

static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
	if (g_isLeftButtonPressed)
	{
		//std::cout << xpos << "," << ypos << std::endl;
		g_pDLSphere->GetLightDirection(g_lightDirection, xpos, ypos);
		g_pShaderProgram->SetDirectionalLight(g_lightDirection);
	}
}


int main()
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	g_pWindow = glfwCreateWindow(g_windowWidth, g_windowHeight, "Face Relit", nullptr, nullptr);
	if (g_pWindow == nullptr)
	{
		cerr << "Failed to create window" << endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(g_pWindow);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize OpenGL context" << std::endl;
		return -1;
	}
	else
	{
		printf("OpenGL %d.%d\n", GLVersion.major, GLVersion.minor);
	}
	glfwSetKeyCallback(g_pWindow, keyCallback);
	glfwSetMouseButtonCallback(g_pWindow, mouseBtnCallback);
	glfwSetCursorPosCallback(g_pWindow, cursorPosCallback);
	

	g_pDLSphere = std::make_unique<DirectionalLightSphere>(g_windowWidth / 2, 0, g_windowWidth/2, g_windowHeight, g_windowWidth, g_windowHeight);

	g_pShaderProgram = std::make_unique<ShaderProgram>();
	g_pShaderProgram->AttachSahder(R"(D:\workspaces\face_relit\Renderer\shader_vertex.glsl)", GL_VERTEX_SHADER);
	g_pShaderProgram->AttachSahder(R"(D:\workspaces\face_relit\Renderer\shader_fragment.glsl)", GL_FRAGMENT_SHADER);
	g_pShaderProgram->Link();
	g_pShaderProgram->Use();
	g_pShaderProgram->SetDefaults();

	std::cout << "Loading Obj File..." << std::endl;
	igl::readOBJ(R"(D:\workspaces\face_relit\face.obj)", V, F);

	std::cout << "Copying Vertices..." << std::endl;
	// copy vertices for updating
	U = V;

	std::cout << "Precomputing Laplace-Beltrami Operator..." << std::endl;
	Utilities::Laplacian::Precompute(V, F, L, &K);

#ifdef NDEBUG
	for (int i = 0; i < 2; i++)
	{
		std::cout << "Smoothing, iteration " << i << "..." << std::endl;
		Utilities::Laplacian::Smooth(U, F, L);
	}

	std::cout << "Computing Normals of Smoothed Mesh..." << std::endl;
#else
	std::cout << "Computing Normals of un-smoothed Mesh..." << std::endl;
#endif
	igl::per_vertex_normals(U, F, N);

	std::cout << "Building Face Model..." << std::endl;
	g_pFaceModel = std::make_unique<FaceModel>(U, N, F, R"(D:\workspaces\face_relit\face_diffuse.png)");

	const auto faceView = glm::lookAt(glm::fvec3{ 96, 96, 300 }, { 96,96,0 }, { 0, -1, 0 });
	const auto facePerspective = glm::perspective<float>(glm::pi<float>() / 4.0f, 1.0f, 0.01f, 1000.0f);

	g_pDLSphere->SetMatrixView(glm::lookAt(glm::fvec3{ 0, 0, 3 }, { 0,0,0 }, { 0, -1, 0 }));
	g_pDLSphere->SetMatrixProjection(glm::ortho<float>(-2, 2, -2, 2, 0.01, 1000));
	
	// all data and state should be ready for rendering
	glfwShowWindow(g_pWindow);

	while (!glfwWindowShouldClose(g_pWindow))
	{
		glfwPollEvents();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);

		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

		glViewport(0, 0, g_windowWidth / 2, g_windowHeight);
		g_pShaderProgram->SetMatrixView(faceView);
		g_pShaderProgram->SetMatrixProjection(facePerspective);
		g_pShaderProgram->SetIsTextured(true);
		g_pFaceModel->Draw();

		glViewport(g_windowWidth / 2, 0, g_windowWidth / 2, g_windowHeight);
		g_pShaderProgram->SetIsTextured(false);
		g_pShaderProgram->SetMatrixView(g_pDLSphere->GetMatrixView());
		g_pShaderProgram->SetMatrixProjection(g_pDLSphere->GetMatrixProjection());
		g_pDLSphere->Draw();

		glfwSwapBuffers(g_pWindow);
	}

	glfwTerminate();

	return 0;
}