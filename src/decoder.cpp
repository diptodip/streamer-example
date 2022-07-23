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


void decoder_process(const char *szInFilePath, int gpu_id, unsigned char* pImage, bool* decoding_flag)
{
    CheckInputFile(szInFilePath);
    std::cout << szInFilePath << std::endl;
    

    CUdeviceptr pTmpImage = 0;
    ck(cuInit(0));
    CUcontext cuContext = NULL;
    createCudaContext(&cuContext, gpu_id, 0);

    FFmpegDemuxer demuxer(szInFilePath);
    NvDecoder dec(cuContext, true, FFmpeg2NvCodecId(demuxer.GetVideoCodec()));
    int nWidth = 0, nHeight = 0;

    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0, iMatrix = 0;
    uint8_t *pVideo = nullptr;
    uint8_t *pFrame;
    
    int frame_i=0;

    do{
        demuxer.Demux(&pVideo, &nVideoBytes);
        nFrameReturned = dec.Decode(pVideo, nVideoBytes);

        if (!nFrame && nFrameReturned)
        {
            LOG(INFO) << dec.GetVideoInfo();
            // Get output frame size from decoder
            nWidth = dec.GetWidth(); nHeight = dec.GetHeight();
            cuMemAlloc(&pTmpImage, nWidth * nHeight * 4);
        }        


        for (int i = 0; i < nFrameReturned; i++) {
            pFrame = dec.GetFrame();
            iMatrix = dec.GetVideoFormatInfo().video_signal_description.matrix_coefficients;
         
            Nv12ToColor32<RGBA32>(pFrame, dec.GetWidth(), (uint8_t*)pTmpImage, 4 * dec.GetWidth(), dec.GetWidth(), dec.GetHeight(), iMatrix);
            GetImage(pTmpImage, pImage, 8 * dec.GetWidth(), dec.GetHeight());

            int counter = 0;
            for (int i = 0; i < 2200; i++) {
                for (int j = 0; j < 3208; j++) {
                    for (int k = 0; k < 4; k++) {
                        printf("%x ", pImage[counter]);
                        counter++;
                    }
                    printf("  ");
                }
                printf("\n");
            }


            // std::string image_frame_name = std::to_string(frame_i) + ".jpg";
            // const char* output_file = image_frame_name.c_str();
            // stbi_write_jpg(output_file, 3208, 2200, 4, pImage, 3208 * 32);
            // frame_i++;
        }	            
        nFrame += nFrameReturned;
    } while (*decoding_flag);
}