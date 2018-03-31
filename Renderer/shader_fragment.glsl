#version 330 core

uniform sampler2D textureDiffuse;
uniform vec3 cameraPosition;
uniform vec3 lightDirection;
uniform bool isTextured;
uniform bool isSpeculared;

struct Material
{
	vec3 Ka;
	vec3 Kd;
	vec3 Ks;
	float Ns;
};

const vec4 Ia = vec4(0.4, 0.4, 0.4, 1.0);

Material defaultMaterial = Material(
	vec3(1.0, 1.0, 1.0),
	vec3(1.0, 1.0, 1.0),
	vec3(0.3, 0.3, 0.3),
	100
);

in vec3 worldPosition;
in vec3 normalInterpolated;
in vec2 texcoord;

out vec4 fragOut;

vec4 calcBlinnPhongLighting(Material M, vec3 LColor, vec3 N, vec3 L, vec3 H)
{
	vec4 Id = vec4(M.Kd * clamp(dot(N, L), 0.0, 1.0), 1.0);

	vec4 Is;
	if (isSpeculared)
		Is = vec4(M.Ks * pow(clamp(dot(N, H), 0.0, 1.0), M.Ns), 0.0);
	else
		Is = vec4(0, 0, 0, 0);

	return (Id + Is) * vec4(LColor, 1);
}

void main()
{
	vec3 N = normalize(normalInterpolated);
	vec3 L = normalize(lightDirection);
	vec3 V = normalize(cameraPosition - worldPosition);
	vec3 H = normalize(L + V);

	//fragOut = vec4(N*0.5+0.5, 1.0);
	//fragOut = vec4(worldPosition.z*0.5 + 0.5, worldPosition.z*0.5 + 0.5, worldPosition.z*0.5 + 0.5, 1.0);
	vec4 Id = vec4(calcBlinnPhongLighting(defaultMaterial, vec3(1, 1, 1), N, L, H));
	vec4 I = Ia + Id;
	if (isTextured)
		fragOut = texture(textureDiffuse, texcoord) * I;
	else
		fragOut = I;

	fragOut /= fragOut.w;
}