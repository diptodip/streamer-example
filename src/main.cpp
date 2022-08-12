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
#include "shader_m.h"

#include <cuda.h>
#include "NvDecoder.h"
#include "NvCodecUtils.h"
#include "FFmpegDemuxer.h"
#include "AppDecUtils.h"
#include "ColorSpace.h"
#include "Logger.h"
#include "gl_helper.h"
#include "decoder.h"
#include "IconsForkAwesome.h"


#include <iostream>       // std::cout
#include <thread>         // std::thread
#include <imfilebrowser.h>

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif


simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();


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
    GLFWwindow* window = glfwCreateWindow(1920, 1080, "Streamer Example", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize OpenGL functions with GLEW
    glew_error_callback(glewInit());



 // ************* Dear Imgui ********************//
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows



    // Setup Dear ImGui style
    ImGui::StyleColorsClassic();

    // When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }


    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load a nice font
    io.Fonts->AddFontFromFileTTF("../fonts/Roboto-Regular.ttf", 15.0f);
    // merge in icons from Font Awesome
    static const ImWchar icons_ranges[] = { ICON_MIN_FK, ICON_MAX_16_FK, 0 };
    ImFontConfig icons_config; icons_config.MergeMode = true; icons_config.PixelSnapH = true;
    io.Fonts->AddFontFromFileTTF("../fonts/forkawesome-webfont.ttf", 15.0f, &icons_config, icons_ranges);
    // use FONT_ICON_FILE_NAME_FAR if you want regular instead of solid


    // Create a OpenGL texture identifier
    GLuint image_texture;
    glGenTextures(1, &image_texture);
    glBindTexture(GL_TEXTURE_2D, image_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 3208, 2200, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same


    // Our state
    ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);


    ImGui::FileBrowser file_dialog;
    file_dialog.SetTitle("title");
    file_dialog.SetTypeFilters({ ".mp4" });

    
    double result = 0.0;
    unsigned long long num_heads = 0;
    unsigned long long num_tails = 0;
    
    
    int size_pic = 3208 * 2200 * 4 *  sizeof(unsigned char);

    // allocate display buffer
    const int size_of_buffer = 32;
    PictureBuffer display_buffer[size_of_buffer];
    for (int i = 0; i < size_of_buffer; i++) {
        display_buffer[i].frame = (unsigned char*)malloc(size_pic);
        display_buffer[i].frame_number = 0;
        display_buffer[i].available_to_write = true;
    }

    std::string input_file;
    std::vector<std::thread> decoder_threads;
    bool* decoding_flag = new bool(false);
    bool* stop_flag = new bool(false);

    int gpu_index = 0;
    int to_display_frame_number = 0;
    int read_head = 0;

    bool play_video = false;
    bool toggle_play_status = false;

    int labeling_frame_number = 0;

    static bool show_app_layout = true;

    SeekContext seek_context;

    int slider_frame_number = 0;
    // Main loop
    while (!glfwWindowShouldClose(window))
    {

        // Poll and handle events (inputs, window resize, etc.)
        glfwPollEvents();


        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

       
        if (ImGui::Begin("File Browser",  NULL, ImGuiWindowFlags_MenuBar))
        {

            if (ImGui::BeginMenuBar())
            {
                if (ImGui::BeginMenu("File"))
                {
                    if (ImGui::MenuItem("Open")) { file_dialog.Open(); };
                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }

            static float f = 0.0f;
            static int counter = 0;
            ImGui::Text("Flip a coin here!");
            ImGui::SameLine();
            if (ImGui::Button("Flip!")) {
                result = ((double)rand() / (RAND_MAX));
                if (result > 0.5) {
                    num_heads++;
                }
                else {
                    num_tails++;
                }
            }
            if (result > 0.5) {
                ImGui::Text("Heads!");
            }
            else {
                ImGui::Text("Tails!");
            }
            if ((num_heads + num_tails) > 0) {
                ImGui::Text("Proportion heads: %.3f", (float)num_heads / (num_heads + num_tails));
            }
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::Text("Frame number %d ", display_buffer[read_head].frame_number); 
        }
        ImGui::End();
        
        file_dialog.Display();

        if (file_dialog.HasSelected())
        {
            input_file = file_dialog.GetSelected().string();
            decoder_threads.push_back(std::thread(&decoder_process, input_file.c_str(), gpu_index, display_buffer, decoding_flag, size_of_buffer, stop_flag, &seek_context));
            file_dialog.ClearSelected();
        }
        
        
        
        if (*decoding_flag) {
            // if the current frame is ready, upload for display, otherwise wait for the frame to get ready 
            while (display_buffer[read_head].frame_number != to_display_frame_number) {
                //std::cout << display_buffer[read_head].frame_number << ", " << to_display_frame_number << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }



        if (play_video) {
            bind_texture(&image_texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 3208, 2200, 0, GL_RGBA, GL_UNSIGNED_BYTE, display_buffer[read_head].frame);
            unbind_texture();
        }


        // for debugging purpose now
        {
            ImGui::SetNextWindowSize(ImVec2(500, 440), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Frames in the buffer", NULL, ImGuiWindowFlags_MenuBar))
            {
                static int selected = 0;
                {
                    for (int i = 0; i < size_of_buffer; i++)
                    {
                        char label[128];
                        sprintf(label, "MyBufferFrame %d", i);
                        if (ImGui::Selectable(label, selected == i)) {
                            selected = (i + read_head) % size_of_buffer;
                            if (!play_video) {
                                bind_texture(&image_texture);
                                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 3208, 2200, 0, GL_RGBA, GL_UNSIGNED_BYTE, display_buffer[selected].frame);
                                unbind_texture();
                            }
                        }
                    }
                }
                ImGui::SameLine();
                ImGui::Text("Frame number %d ", display_buffer[selected].frame_number);

            }
            ImGui::End();
        }


        if (toggle_play_status && play_video) {
            play_video = false;
            toggle_play_status = false;
        }


        // Render a video frame
        {

            ImGui::Begin("Camera!");
           
            ImGui::BeginGroup();
            ImGui::BeginChild("scene view", ImVec2(0, -ImGui::GetFrameHeightWithSpacing())); // Leave room for 1 line below us

            ImVec2 avail_size = ImGui::GetContentRegionAvail();
            ImGui::Image((void*)(intptr_t)image_texture, avail_size);
            ImGui::EndChild();

            if (ImGui::Button(play_video ? ICON_FK_PAUSE : ICON_FK_PLAY))
            {
                play_video = !play_video;
            }


            ImGui::SameLine();
            // Arrow buttons with Repeater
            float spacing = ImGui::GetStyle().ItemInnerSpacing.x;
            ImGui::PushButtonRepeat(true);
            if (ImGui::Button(ICON_FK_MINUS)) { labeling_frame_number--; }
            ImGui::SameLine(0.0f, spacing);
            if (ImGui::Button(ICON_FK_PLUS)) {
                // advance_clicks++;
                play_video = true;
                toggle_play_status = true;
            }
            ImGui::PopButtonRepeat();
            ImGui::SameLine();

            ImGui::SliderInt("##frame count", &slider_frame_number, 1, 950);

            //std::cout << to_display_frame_number << std::endl;
            //ImGui::SliderInt("##test slider", &test_slider, 1, 3000);

            if (ImGui::IsItemDeactivatedAfterEdit()) {
                std::cout << "slider frame number: " << slider_frame_number << std::endl;
                // change to seek to closest keyframe 


                seek_context.seek_frame = (uint64_t)slider_frame_number;
                std::cout << "convert seek frame in main: " << seek_context.seek_frame << std::endl;
                seek_context.use_seek = true;
            
                while (seek_context.use_seek) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                to_display_frame_number = seek_context.seek_frame;
                read_head = 0;
                *decoding_flag = false;
                play_video = false;
                std::cout << "main thread: " << to_display_frame_number << std::endl;
            }

            ImGui::EndGroup();
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
        
        
        // Update and Render additional Platform Windows
        // (Platform functions may change the current OpenGL context, so we save/restore it to make it easier to paste this code elsewhere.
        //  For this specific demo app we could also call glfwMakeContextCurrent(window) directly)
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(window);
 
        
        if(*decoding_flag && play_video){
            to_display_frame_number++;
            display_buffer[read_head].available_to_write = true;
            read_head = (read_head + 1) % size_of_buffer;
            //slider_frame_number = to_display_frame_number;
        }
        
       
    }


    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    *stop_flag = true;
    // wait for threads to join
    for (auto& t : decoder_threads)
        t.join();

    return 0;
}
