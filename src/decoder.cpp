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


void decoder_process(const char *input_file_name, int gpu_id, PictureBuffer* display_buffer, bool* decoding_flag, int size_of_buffer, bool* stop_flag, SeekContext* seek_context)
{
    CheckInputFile(input_file_name);
    std::cout << input_file_name << std::endl;
    
    CUdeviceptr pTmpImage = 0;
    ck(cuInit(0));
    CUcontext cuContext = NULL;
    createCudaContext(&cuContext, gpu_id, 0);

    std::map<std::string, std::string> m;
    size_t nVideoBytes = 0;
    PacketData pktinfo;


    FFmpegDemuxer demuxer(input_file_name, m);
    NvDecoder dec(cuContext, true, FFmpeg2NvCodecId(demuxer.GetVideoCodec()));
    int nWidth = 0, nHeight = 0;

    int nFrameReturned = 0, nFrame = 0, iMatrix = 0;
    uint8_t *pVideo = nullptr;
    uint8_t *pFrame;
    
    int buffer_head=0;

    uint64_t decoder_frame_num;

    do{

        if (seek_context->use_seek) {

            std::cout << "target_frame_number:" << seek_context->seek_frame << std::endl;
            demuxer.Seek(*seek_context, pVideo, nVideoBytes, pktinfo);
            decoder_frame_num = demuxer.FrameNumberFromTs(pktinfo.dts);
            std::cout << "frame_number:" << decoder_frame_num << std::endl;

            // reset the display buffer after seeking  
            buffer_head = 0;
            for (int i = 0; i < size_of_buffer; i++) {
                display_buffer[i].available_to_write = true;
            }
            nFrame = decoder_frame_num;
            seek_context->use_seek = false;
        }
        else {
            demuxer.Demux(pVideo, nVideoBytes, pktinfo); 
        }

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
                while (!display_buffer[buffer_head].available_to_write && !(*stop_flag)) {
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
    } while (!(*stop_flag));
}