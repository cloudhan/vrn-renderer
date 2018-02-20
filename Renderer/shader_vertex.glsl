#version 330 core

layout(location = 0) in vec3 vertex;
layout(location = 1) in vec3 normal;

out vec3 worldPosition;
out vec3 normalInterpolated;
out vec2 texcoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 perspective;

void main()
{
	worldPosition = vec3(model * vec4(vertex, 1.0));
	gl_Position = perspective * view * vec4(worldPosition, 1.0);
	normalInterpolated = normalize(normal);
	texcoord = vec2(vertex.x / 192.0, vertex.y / 192.0);
}