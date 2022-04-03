// Example of streaming images computed on a GPU with CUDA to a PBO in
// OpenGL, rendered with DearImgui. Structure follows basic OpenGL
// example using the ImGui GLFW backend.

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <GL/glew.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "create_image_cuda.h"
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

// [Win32] The Dear ImGui example includes a copy of glfw3.lib
// pre-compiled with VS2010 to maximize ease of testing and
// compatibility with old VS compilers.  To link with VS2010-era
// libraries, VS2015+ requires linking with
// legacy_stdio_definitions.lib, which we do using this pragma.  Your
// own project should not be affected, as you are likely to link with
// a newer binary of GLFW that is adequate for your version of Visual
// Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

static void glew_error_callback(GLenum glew_error)
{
	if (GLEW_OK != glew_error) {
		printf("GLEW error: %s\n", glewGetErrorString(glew_error));
	}
}

static void create_texture(GLuint *texture) {
        // Create a OpenGL texture identifier
	glGenTextures(1, texture);
	glBindTexture(GL_TEXTURE_2D, *texture);

	// Setup filtering parameters for display
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

static void bind_texture(GLuint *texture)
{
	glBindTexture(GL_TEXTURE_2D, *texture);
}

static void unbind_texture()
{
	glBindTexture(GL_TEXTURE_2D, 0);
}

static void create_pbo(GLuint *pbo)
{
	glGenBuffers(1, pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 854 * 480 * 4 * sizeof(unsigned char), 0, GL_STREAM_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

static void bind_pbo(GLuint *pbo)
{
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
}

static void unbind_pbo()
{
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

static void register_pbo_to_cuda(GLuint *pbo, cudaGraphicsResource_t *cuda_resource)
{
	cudaGraphicsGLRegisterBuffer(cuda_resource, *pbo, cudaGraphicsRegisterFlagsNone);
}

static void map_cuda_resource(cudaGraphicsResource_t *cuda_resource)
{
	cudaGraphicsMapResources(1, cuda_resource);
}

static void cuda_pointer_from_resource(unsigned char **cuda_buffer_p, size_t *size_p, cudaGraphicsResource_t *cuda_resource)
{
	cudaGraphicsResourceGetMappedPointer((void **) cuda_buffer_p, size_p, *cuda_resource);
}

static void unmap_cuda_resource(cudaGraphicsResource_t *cuda_resource)
{
	cudaGraphicsUnmapResources(1, cuda_resource);
}

static void upload_image_pbo_to_texture()
{
	// Assume PBO is bound before this, therefore the last
	// argument is an offset into the PBO, not a pointer to a
	// buffer stored in CPU memory
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 854, 480, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
}

int main(int, char**)
{
	// Setup window
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit())
	    return 1;

	// Decide GL+GLSL versions
#if defined(__APPLE__)
	// GL 3.2 + GLSL 150
	const char* glsl_version = "#version 150";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
	// GL 3.0 + GLSL 130
	const char* glsl_version = "#version 130";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#endif

	// Create window with graphics context
	GLFWwindow* window = glfwCreateWindow(1280, 720, "Streamer Example", NULL, NULL);
	if (window == NULL)
	    return 1;
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1); // Enable vsync

	// Initialize OpenGL functions with GLEW
	glew_error_callback(glewInit());

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	// Setup Dear ImGui style
	ImGui::StyleColorsClassic();

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	// Load a nice font
	io.Fonts->AddFontFromFileTTF("Roboto-Regular.ttf", 15.0f);

	// Our state
	ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);
	double result = 0.0;
	unsigned long long num_heads = 0;
	unsigned long long num_tails = 0;
	GLuint texture;
	GLuint pbo;
	cudaGraphicsResource_t cuda_resource = 0;
	unsigned char *cuda_buffer;
	size_t cuda_pbo_storage_buffer_size;
	create_texture(&texture);
	create_pbo(&pbo);
	bind_pbo(&pbo);
	register_pbo_to_cuda(&pbo, &cuda_resource);
	unbind_texture();
	unbind_pbo();


	// Main loop
	while (!glfwWindowShouldClose(window))
	{
	    // Poll and handle events (inputs, window resize, etc.)
	    glfwPollEvents();
	    // Create image on CUDA and transfer to PBO then OpenGL texture
	    // CUDA-GL INTEROP STARTS HERE -------------------------------------------------------------------------
	    map_cuda_resource(&cuda_resource);
	    cuda_pointer_from_resource(&cuda_buffer, &cuda_pbo_storage_buffer_size, &cuda_resource);
	    create_image_cuda(cuda_buffer); // CUDA computation of an image!
	    unmap_cuda_resource(&cuda_resource);
	    // CUDA-GL INTEROP ENDS HERE ---------------------------------------------------------------------------
	    bind_pbo(&pbo);
	    bind_texture(&texture);
	    upload_image_pbo_to_texture(); // Needs no arguments because texture and PBO are bound
	    unbind_texture();
	    unbind_pbo();

	    // Start the Dear ImGui frame
	    ImGui_ImplOpenGL3_NewFrame();
	    ImGui_ImplGlfw_NewFrame();
	    ImGui::NewFrame();

	    // Show a simple window that we create ourselves. This is
	    // just to show that tracking of state is working
	    // separately from CUDA.
	    {
		static float f = 0.0f;
		static int counter = 0;
		ImGui::SetNextWindowSize(ImVec2(0, 0), 0); // Setting size to 0, 0 forces auto-fit
		ImGui::Begin("Hello, world!");
		ImGui::Text("Flip a coin here!");
		ImGui::SameLine();
		if (ImGui::Button("Flip!")) {
			result = ((double) rand() / (RAND_MAX));
			if (result > 0.5) {
				num_heads++;
			} else {
				num_tails++;
			}
		}
		if (result > 0.5) {
			ImGui::Text("Heads!");
		} else {
			ImGui::Text("Tails!");
		}
		if ((num_heads + num_tails) > 0) {
			ImGui::Text("Proportion heads: %.3f", (float) num_heads / (num_heads + num_tails));
		}
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		ImGui::End();
	    }

	    // Render a video frame
	    {
		ImGui::SetNextWindowSize(ImVec2(0, 0), 0); // Setting size to 0, 0 forces auto-fit
		ImGui::Begin("Hello, video!");
		ImGui::Text("pointer = %p", texture);
		ImGui::Image((void*)(intptr_t)texture, ImVec2(854, 480));
		ImGui::End();
	    }

	    // Rendering
	    ImGui::Render();
	    int display_w, display_h;
	    glfwGetFramebufferSize(window, &display_w, &display_h);
	    glViewport(0, 0, display_w, display_h);
	    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
	    glClear(GL_COLOR_BUFFER_BIT);
	    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	    glfwSwapBuffers(window);
	}

	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}
