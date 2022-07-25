#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION

#include "decoder.h"

void GetImage(CUdeviceptr dpSrc, uint8_t *pDst, int nWidth, int nHeight)
{
    CUDA_MEMCPY2D m = { 0 };
    m.WidthInBytes = nWidth;
    m.Height = nHeight;
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = (CUdeviceptr)dpSrc;
    m.srcPitch = m.WidthInBytes;
    m.dstMemoryType = CU_MEMORYTYPE_HOST;
    m.dstDevice = (CUdeviceptr)(m.dstHost = pDst);
    m.dstPitch = m.WidthInBytes;
    cuMemcpy2D(&m);
}


void decoder_process(const char *input_file_name, int gpu_id, PictureBuffer* display_buffer, bool* decoding_flag, int size_of_buffer)
{
    CheckInputFile(input_file_name);
    std::cout << input_file_name << std::endl;
    

    CUdeviceptr pTmpImage = 0;
    ck(cuInit(0));
    CUcontext cuContext = NULL;
    createCudaContext(&cuContext, gpu_id, 0);

    FFmpegDemuxer demuxer(input_file_name);
    NvDecoder dec(cuContext, true, FFmpeg2NvCodecId(demuxer.GetVideoCodec()));
    int nWidth = 0, nHeight = 0;

    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0, iMatrix = 0;
    uint8_t *pVideo = nullptr;
    uint8_t *pFrame;
    
    int buffer_head=0;

    do{
        demuxer.Demux(&pVideo, &nVideoBytes);
        nFrameReturned = dec.Decode(pVideo, nVideoBytes);

        if (!nFrame && nFrameReturned)
        {
            LOG(INFO) << dec.GetVideoInfo();
            // Get output frame size from decoder
            nWidth = dec.GetWidth(); nHeight = dec.GetHeight();
            int size_in_bytes = nWidth * nHeight * 4;
            cuMemAlloc(&pTmpImage, size_in_bytes);
        }        


        for (int i = 0; i < nFrameReturned; i++) {
            // decode frame and conversion
            pFrame = dec.GetFrame();
            iMatrix = dec.GetVideoFormatInfo().video_signal_description.matrix_coefficients;
            Nv12ToColor32<RGBA32>(pFrame, dec.GetWidth(), (uint8_t*)pTmpImage, 4 * dec.GetWidth(), dec.GetWidth(), dec.GetHeight(), iMatrix);
            
            if (nFrame == 0) {
                GetImage(pTmpImage, display_buffer[buffer_head].frame, 4 * dec.GetWidth(), dec.GetHeight());
                display_buffer[buffer_head].available_to_write = false;
                *decoding_flag = true;
                display_buffer[buffer_head].frame_number = nFrame;
            }
            else {
                while (!display_buffer[buffer_head].available_to_write) {
                    // if the next frame hasn't been displayed, the queue is full, sleep  
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                GetImage(pTmpImage, display_buffer[buffer_head].frame, 4 * dec.GetWidth(), dec.GetHeight());
                display_buffer[buffer_head].available_to_write = false;
                display_buffer[buffer_head].frame_number = nFrame;
            }
            
            nFrame = nFrame + 1;
            buffer_head = (buffer_head + 1) % size_of_buffer;
        }	            
    } while (true);
}