// Tested with Substance 2.5.3.
//
// A preview GLSL shader for the "Super" shader. Written by IanBanks on the Bohemia
// Interactive Forums.
//
// This shader may only be used within Allegorithmic 
// products (https://www.allegorithmic.com/) for the purpose of producing assets
// for use within games sold by Bohemia Interactive (https://www.bohemia.net/).
//
// The channels used are:
// 
//     Base Color (_co) [sRGB]
//     Height (_nohq) [L]
//     Specular level (_smdi) [L]
//     Glossiness (_smdi) [L]
//     User0 (Ambient Occlusion, _as green channel, Optional) [L]
//     User1 (Diffuse Occlusion, _as blue channel, Optional) [L]
//     User2 (Macro, Optional) [RGB]
//     User3 (Detail, Optional) [RGB]
//     User4 (Macro Alpha, Optional) [L]
//
// The environment should also be set to a regular ARMA 3 environment map, not a third party HDR.

import lib-defines.glsl
import lib-env.glsl
import lib-normal.glsl
import lib-random.glsl

//: param custom {
//:   "default": 40.0,
//:   "label": "Specular power",
//:   "min": 0.0,
//:   "max": 1500.0
//: }
uniform float specular_power_constant;

//: param custom { "default": 1.0, "label": "Ambient", "widget": "color" }
uniform vec3 ambient_constant;

//: param custom { "default": 0.25, "label": "Specular", "widget": "color" }
uniform vec3 specular_constant;

//: param custom { "default": 0.0, "label": "Forced diffuse", "widget": "color" }
uniform vec3 forced_diffuse_constant;

//: param custom { "default": 1.0, "label": "Diffuse", "widget": "color" }
uniform vec3 diffuse_constant;

//: param custom { "default": 0.0, "label": "Emissive", "widget": "color" }
uniform vec3 emissive_constant;

//: param custom {
//:   "default": 0.0,
//:   "label": "Emissive Scale",
//:   "min": 0.0,
//:   "max": 20.0
//: }
uniform float emissive_scale_constant;

//: param custom {
//:   "default": 1.3,
//:   "label": "Fresnel N",
//:   "min": 0.01,
//:   "max": 10.0
//: }
uniform float fresnel_n_constant;

//: param custom {
//:   "default": 7,
//:   "label": "Fresnel K",
//:   "min": 0.05,
//:   "max": 10.0
//: }
uniform float fresnel_k_constant;

//: param auto environment_max_lod
uniform float environment_max_lod;

//: param auto channel_specularlevel
uniform sampler2D specularlevel_tex;

//: param auto channel_glossiness
uniform sampler2D glossiness_tex;

//: param auto channel_basecolor
uniform sampler2D basecolor_tex;

//: param auto channel_user0
uniform sampler2D ambient_shadow_green_tex;

//: param auto channel_user0_is_set
uniform bool ambient_shadow_green_is_set;

//: param auto channel_user1
uniform sampler2D ambient_shadow_blue_tex;

//: param auto channel_user1_is_set
uniform bool ambient_shadow_blue_is_set;

//: param auto channel_user2
uniform sampler2D macro_tex;

//: param auto channel_user2_is_set
uniform bool macro_is_set;

//: param auto channel_user3
uniform sampler2D detail_tex;

//: param auto channel_user3_is_set
uniform bool detail_is_set;

//: param auto channel_user4
uniform sampler2D macro_alpha_tex;

//: param auto channel_user4_is_set
uniform bool macro_alpha_is_set;

//: param auto world_eye_position
uniform vec3 uniform_world_eye_position;

//: param custom {
//:   "default": 0,
//:   "label": "Preview Mode",
//:   "widget": "combobox",
//:   "values": {
//:     "Material (Filmic Tone Mapping)": 0,
//:     "Material (Reinhard Tone Mapping)": 1,
//:     "Material (Disable Tone Mapping)": 2,
//:     "Specular with Fresnel": 3,
//:     "Fresnel": 4,
//:     "Incidence (Camera)": 5,
//:     "Normals (World)": 6,
//:     "Environment Reflection": 7
//:   }
//: }
uniform int preview_mode;

//: param custom {
//:   "default": 0,
//:   "label": "Lighting",
//:   "widget": "combobox",
//:   "values": {
//:     "Apex Midday": 0,
//:     "Apex Sunrise": 1,
//:     "Apex Twilight": 2,
//:     "VBS Noon": 3
//:   }
//: }
uniform int lighting_mode;

//: param custom {
//:   "default": 0,
//:   "label": "Environment Source",
//:   "widget": "combobox",
//:   "values": {
//:     "Substance Painter (_co suffix)": 0,
//:     "HDR (_ca suffix)": 1
//:   }
//: }
uniform int environment_source;

//: param custom { "default": "", "default_color": [1.0, 1.0, 1.0, 1.0], "label": "HDR Environment Map" }
uniform sampler2D hdr_environment_texture;

struct MapLighting
{
	vec4 diffuse;
	vec4 ambient;
	vec4 ambientMid;
	vec4 groundReflection;
	vec3 bidirect;
	float aperture;
};

MapLighting getVbsMapLighting(vec4 diffuse, vec4 ambient, vec3 vbs_ground_reflection)
{
	vec4 armaGroundReflection = ambient * vec4(vbs_ground_reflection, 1.0);

	return MapLighting( 
		diffuse,
		ambient,
		0.5 * (ambient + armaGroundReflection),
		armaGroundReflection,
		vbs_ground_reflection,
		60.0);
}

MapLighting getMapLighting()
{
	switch (lighting_mode)
	{
		case 0:
			// Sun angle 90, overcast 0:
			return MapLighting(
				vec4(1, 0.87, 0.85, 17.2),
				vec4(0.498, 0.602, 0.77, 14.8),
				vec4(0.635, 0.635, 0.663, 14.504),
				vec4(0.745, 0.671, 0.643, 14.21392),
				vec3(0.025, 0.024, 0.018),
				120.0);

		case 1:
			// Sun angle 2, overcast 0:
			return MapLighting( 
				vec4(0.95, 0.42, 0.22, 8.4),
				vec4(0.306, 0.357, 0.463, 8.4),
				vec4(0.365, 0.361, 0.396, 7.392),
				vec4(0.416, 0.38, 0.388, 7.09632),
				vec3(0.023, 0.024, 0.025),
				10.0);

		case 2:
			// Sun angle -5, overcast 0:
			return MapLighting( 
				vec4(0.16, 0.18, 0.28, 3), 
				vec4(0.173, 0.239, 0.373, 4.6), 
				vec4(0.173, 0.239, 0.373, 4.048), 
				vec4(0.173, 0.239, 0.373, 3.88608), 
				vec3(0.0115, 0.012, 0.0125), 
				6.0);

		case 3:
			// VBS noon:
			return getVbsMapLighting(
				vec4(1, 1, 1, 17 - 4),
				vec4(1, 1.3, 1.55, 13.5 - 4),
				vec3(0.085, 0.068, 0.034));
	}
}

const vec3 luminosity = vec3(0.299, 0.587, 0.114);
const vec3 reinhard_tonemap_luminosity = vec3(0.2126, 0.7152, 0.0722);

vec3 mapColourToLinearColour(float apertureFactor, vec4 map_colour)
{
	float luminosity_scaling = pow(2.0, map_colour.a) / dot(luminosity, map_colour.rgb);

	return apertureFactor * luminosity_scaling * map_colour.rgb;
}

struct LightingInputs
{
	vec3 PSC_Diffuse;
	vec3 PSC_DForced;

	vec3 PSC_AE;
	vec3 PSC_AmbientMid;
	vec3 PSC_GE;

	vec3 PSC_GlassMatSpecular;
	vec3 PSC_Emissive;
	vec3 PSC_GlassEnvColor;
	vec3 PSC_LDirectionGround_DiffuseBack;
	vec3 PSC_Specular;
};

LightingInputs getInputsFromMapLighting(MapLighting map)
{
	float apertureFactor = pow(1.0 / map.aperture, 2.0);

	LightingInputs inputs;

	inputs.PSC_Diffuse = 0.5 * diffuse_constant * mapColourToLinearColour(apertureFactor, map.diffuse);
	inputs.PSC_DForced = 0.5 * forced_diffuse_constant * mapColourToLinearColour(apertureFactor, map.diffuse);

	inputs.PSC_AE = 0.5 * ambient_constant * mapColourToLinearColour(apertureFactor, map.ambient);
	inputs.PSC_AmbientMid = 0.5 * ambient_constant * mapColourToLinearColour(apertureFactor, map.ambientMid);
	inputs.PSC_GE = 0.5 * ambient_constant * mapColourToLinearColour(apertureFactor, map.groundReflection);

	inputs.PSC_GlassMatSpecular = 0.5 * specular_constant;

	// Emissive scale constant is just a work-around for Substance not making HDR colours easy to set:
	inputs.PSC_Emissive = 0.5 * apertureFactor * emissive_constant * pow(2.0, emissive_scale_constant);

	const float glass_env_diffuse_factor = 0.05;
	inputs.PSC_GlassEnvColor = 0.5 * (glass_env_diffuse_factor * mapColourToLinearColour(apertureFactor, map.diffuse) +
		mapColourToLinearColour(apertureFactor, map.ambient));

	inputs.PSC_LDirectionGround_DiffuseBack = map.bidirect * inputs.PSC_Diffuse;

	float specular_scale = specular_power_constant > 0.0 ?
		(0.0761544 / pow(acos(pow(0.5, 1.0 / specular_power_constant)), 2.0)) : 1.0;

	inputs.PSC_Specular = specular_scale * mapColourToLinearColour(apertureFactor, map.diffuse) * specular_constant;

	return inputs;
}

LightingInputs lighting = getInputsFromMapLighting(getMapLighting());

vec4 environmentSampleSelectedMap(vec2 sample_coordinates, float specular_power_lod)
{
	switch (environment_source)
	{
		case 0:
			return vec4(textureLod(environment_texture, sample_coordinates, specular_power_lod).xyz, 1.0);

		case 1:
			return textureLod(hdr_environment_texture, sample_coordinates, specular_power_lod);
	}
}

vec3 environmentSample(vec3 world_reflection, vec2 tex_coord, float specular_exponent)
{
	float maximum_specular_power = 1000.0;
	float specular_power_to_mip_exponent = 10.0;
	float maximum_specular_power_mip = environment_max_lod * 0.8;

	float scaled_specular_power = 1.0 - clamp(
		specular_exponent / maximum_specular_power, 0.0, 1.0);

	float specular_power_lod = maximum_specular_power_mip * 
		pow(scaled_specular_power, specular_power_to_mip_exponent);

	vec2 sample_coordinates = 0.5 + vec2(0.5, 0.5) * world_reflection.xy;

	vec4 environment_sample = environmentSampleSelectedMap(sample_coordinates, specular_power_lod);

	vec3 linear_sample = pow(environment_sample.rgb, vec3(2.2));

	vec3 hdr_sample = linear_sample * pow(2.0, 4.0 * (1.0 - environment_sample.a));

	return lighting.PSC_GlassEnvColor * hdr_sample;
}

float fresnelSample(float u)
{
	float n = fresnel_n_constant;
	float k = fresnel_k_constant;

	float n_squared = n * n;
	float k_squared = k * k;

	float s = acos(u);
	float sin_s_squared = pow(sin(s), 2.0);

	float aa_bb_right = n_squared - k_squared - sin_s_squared;
	float aa_bb_left = sqrt(pow(aa_bb_right, 2.0) + 4.0 * n_squared * k_squared);

	float aa_squared = (aa_bb_left + aa_bb_right) / 2.0;
	float bb_squared = (aa_bb_left - aa_bb_right) / 2.0;

	float aa = sqrt(aa_squared);

	float fs = 
		(aa_squared + bb_squared - 2.0 * aa * u + pow(u, 2.0)) / 
		(aa_squared + bb_squared + 2.0 * aa * u + pow(u, 2.0));

	float fp = fs * (aa_squared + bb_squared - 2.0 * aa * sin(s) * tan(s) + sin_s_squared * pow(tan(s), 2.0)) /
		(aa_squared + bb_squared + 2.0 * aa * sin(s) * tan(s) + sin_s_squared * pow(tan(s), 2.0));

	return (fs + fp) / 2.0;
}

const float map_tonemapShoulderStrength = 0.22;
const float map_tonemapLinearStrength = 0.12;
const float map_tonemapLinearAngle = 0.1;
const float map_tonemapToeStrength = 0.2;
const float map_tonemapToeNumerator = 0.022;
const float map_tonemapToeDenominator = 0.2;
const float map_tonemapLinearWhite = 11.2;
const float map_tonemapExposureBias = 1.0;

vec3 tonemapFunction(vec3 x)
{
	const float a = map_tonemapShoulderStrength;
	const float b = map_tonemapLinearStrength;
	const float c = map_tonemapLinearAngle;
	const float d = map_tonemapToeStrength;
	const float e = map_tonemapToeNumerator;
	const float f = map_tonemapToeDenominator;

	return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f;
}

float tonemapDenominator = tonemapFunction(vec3(map_tonemapLinearWhite)).x;

vec3 applyFilmicTonemap(vec3 x)
{
	return tonemapFunction(x * map_tonemapExposureBias) / tonemapDenominator;
}

const float map_tonemapReinhardWhitepoint = 0.9;

vec3 applyReinhardTonemap(vec3 x)
{
	float luminosity = dot(reinhard_tonemap_luminosity, x);

	float scaleNumerator = luminosity * (1.0 + luminosity / pow(map_tonemapReinhardWhitepoint, 2.0));
	float scale = (scaleNumerator / (luminosity + 1)) / (luminosity + 0.0001);

	return clamp(scale * x, 0.0, 1.0);
}

vec4 shade(V2F inputs) 
{
	vec3 tangent_base_normal = normalUnpack(texture(base_normal_texture, inputs.tex_coord), base_normal_y_coeff);
	vec3 tangent_height_normal = normalFromHeight(inputs.tex_coord, height_force);
	vec3 tangent_blended_normal = normalBlend(tangent_base_normal, tangent_height_normal);

	vec3 world_normal = normalize(
		tangent_blended_normal.x * inputs.tangent +
		tangent_blended_normal.y * inputs.bitangent +
		tangent_blended_normal.z * inputs.normal
	);

	vec3 world_eye = normalize(uniform_world_eye_position - inputs.position);

	vec3 world_reflection = reflect(-world_eye, world_normal);

	float specular_channel = texture(specularlevel_tex, inputs.tex_coord).x;

	float eye_incidence = dot(world_normal, world_reflection);
	float raw_fresnel = fresnelSample(eye_incidence);
	float fresnel = specular_channel * raw_fresnel;

	float ambient_gradient_direction = world_normal.y;

	vec3 top_colour = mix(lighting.PSC_AmbientMid, lighting.PSC_AE, clamp(ambient_gradient_direction, 0.0, 1.0));
	vec3 bottom_colour = mix(lighting.PSC_GE, lighting.PSC_AmbientMid, 1.0 + clamp(ambient_gradient_direction, -1.0, 0.0));

	vec3 ambient_and_emissive = lighting.PSC_Emissive + mix(bottom_colour, top_colour, 
		sign(ambient_gradient_direction) * 0.5 + 0.5);

	vec3 world_light = normalize(
		vec3(cos(environment_rotation * M_2PI), 1, sin(environment_rotation * M_2PI)));
	vec3 world_half = normalize(world_eye + world_light);
	float normal_dot_light = dot(world_normal, world_light);

	float normal_dot_half = dot(world_normal, world_half);

	float specular_power_channel = texture(glossiness_tex, inputs.tex_coord).x;
	float specular_exponent = specular_power_constant * specular_power_channel;

	float diffuse_lit_coefficient = max(normal_dot_light, 0.0);
	float specular_lit_coefficient = normal_dot_light > 0.0 ? pow(max(0.0, normal_dot_half), specular_exponent) : 0.0;

	specular_lit_coefficient = min(1.0, specular_lit_coefficient);

	vec3 back_facing_diffuse = lighting.PSC_LDirectionGround_DiffuseBack * max(0.0, -normal_dot_light);

	// This also appears to be zero based on Buldozer testing:
	vec3 unknown_ambient = vec3(0, 0, 0);

	float ambient_shadow_channel =
		ambient_shadow_green_is_set ? texture(ambient_shadow_green_tex, inputs.tex_coord).x : 1.0;

	vec3 output_indirect = unknown_ambient + (ambient_and_emissive + back_facing_diffuse) *
		ambient_shadow_channel;

	vec3 output_direct = lighting.PSC_DForced + lighting.PSC_Diffuse * diffuse_lit_coefficient * (1.0 - fresnel);
	vec3 output_specular = lighting.PSC_Specular * specular_lit_coefficient * fresnel;
	vec3 environment_value = environmentSample(world_reflection, inputs.tex_coord, specular_exponent);
	vec3 output_specular_environment = lighting.PSC_GlassMatSpecular * 2.0 * fresnel *
		environment_value;

	vec3 base_sample = texture(basecolor_tex, inputs.tex_coord).xyz;
	vec4 macro_sample =
		macro_is_set ? texture(macro_tex, inputs.tex_coord) : vec4(0, 0, 0, 0);
	float macro_alpha_sample = 
		macro_alpha_is_set ? texture(macro_alpha_tex, inputs.tex_coord).x : 0.0;
	vec3 detail_sample = detail_is_set ? texture(detail_tex, inputs.tex_coord).rgb : vec3(0.5, 0.5, 0.5);

	vec3 output_colour = mix(base_sample, macro_sample.rgb, macro_alpha_sample) * 2.0 * detail_sample;

	float direct_shadow_channel =
		ambient_shadow_blue_is_set ? texture(ambient_shadow_blue_tex, inputs.tex_coord).x : 1.0;

	vec3 non_specular_result = (output_indirect + output_direct * direct_shadow_channel) * output_colour;
	vec3 specular_result = output_specular * direct_shadow_channel + output_specular_environment;

	vec3 material_result = non_specular_result + specular_result;

	switch (preview_mode)
	{
		case 0:
			return vec4(applyFilmicTonemap(material_result), 1.0);

		case 1:
			return vec4(applyReinhardTonemap(material_result), 1.0);

		case 2:
			return vec4(material_result, 1.0);

		case 3:
			return vec4(vec3(specular_lit_coefficient * fresnel), 1.0);

		case 4:
			return vec4(vec3(raw_fresnel), 1.0);

		case 5:
			return vec4(vec3(eye_incidence), 1.0);

		case 6:
			return vec4(world_normal, 1.0);

		case 7:
			return vec4(environment_value, 1.0);
	}
}

