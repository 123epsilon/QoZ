#ifndef SZ3_SZINTERP_HPP
#define SZ3_SZINTERP_HPP

#include "QoZ/compressor/SZInterpolationCompressor.hpp"

#include "QoZ/compressor/deprecated/SZBlockInterpolationCompressor.hpp"

#include "QoZ/preprocessor/Wavelet.hpp"

#include "QoZ/quantizer/IntegerQuantizer.hpp"
#include "QoZ/lossless/Lossless_zstd.hpp"
#include "QoZ/utils/Iterator.hpp"
#include "QoZ/utils/Sample.hpp"
#include "QoZ/utils/Transform.hpp"
#include "QoZ/utils/Statistic.hpp"
#include "QoZ/utils/Extraction.hpp"
#include "QoZ/utils/QuantOptimization.hpp"
#include "QoZ/utils/Config.hpp"
#include "QoZ/utils/Metrics.hpp"
#include "QoZ/utils/CoeffRegression.hpp"
#include "QoZ/utils/ExtractRegData.hpp"
#include "QoZ/api/impl/SZLorenzoReg.hpp"

#include "QoZ/sperr/SPERR3D_OMP_C.h"

#include "QoZ/sperr/SPERR3D_OMP_D.h"


//#include <pybind11/embed.h>
//#include <pybind11/numpy.h>
//#include <pybind11/stl.h>

//#include <cunistd>
#include <cmath>
#include <memory>
#include <limits>
#include <cstring>
#include <cstdlib>
//namespace py = pybind11;


template<class T, QoZ::uint N>
bool use_sperr(const QoZ::Config & conf){
    return ( (conf.wavelet>0 or conf.sperrWithoutWave) and conf.sperr>=conf.wavelet and N==3);
}

template<class T, QoZ::uint N>
auto pre_Condition(const QoZ::Config &conf,T * data){//conditioner not updated to the newest version of SPERR.
    std::vector<double> buf(data,data+conf.num);//maybe not efficient
    sperr::Conditioner conditioner;
    sperr::dims_type temp_dims={0,0,0};//temp. Fix for later introducing custom filter.
    /*
    if(conf.conditioning==2){
        std::array<bool, 4> b4{true,true,false,false};
        conditioner.toggle_all_settings(b4);
    }
    */

    auto condi_meta = conditioner.condition(buf,temp_dims);
    //if(rtn!=sperr::RTNType::Good)
        //std::cout<<"bad cond"<<std::endl;
    for(size_t i=0;i<conf.num;i++)
        data[i]=buf[i];
    //memcpy(data,buf.data(),conf.num*sizeof(T));//maybe not efficient
    return condi_meta;
}

template<class T, QoZ::uint N>
auto post_Condition(T * data,const size_t &num,const sperr::vec8_type& meta){
    std::vector<double> buf(data,data+num);
   
    sperr::dims_type temp_dims={0,0,0};//temp. Fix for later introducing custom filter.
    sperr::Conditioner conditioner;
    auto rtn = conditioner.inverse_condition(buf,temp_dims,meta);
    for(size_t i=0;i<num;i++)
        data[i]=buf[i];
    //memcpy(data,buf.data(),num*sizeof(T));//maybe not efficient
    return rtn;
}

template<class T, QoZ::uint N> 
char *SPERR_Compress(QoZ::Config &conf, T *data, size_t &outSize){
    assert(N==2 or N==3);//need to complete 2D support later.
        
    SPERR3D_OMP_C compressor;
    compressor.set_num_threads(1);
    compressor.set_eb_coeff(conf.wavelet_rel_coeff);
    if(conf.wavelet!=1)
        compressor.set_skip_wave(true);
    auto rtn = sperr::RTNType::Good;
      
    auto chunks = std::vector<size_t>{1024,1024,1024};//ori 256^3, to tell the truth this is not large enough for scale but I just keep it, maybe set it large later.
    if(N==3)
        rtn = compressor.copy_data(reinterpret_cast<const float*>(data), conf.num,
                                {conf.dims[2], conf.dims[1], conf.dims[0]}, {chunks[0], chunks[1], chunks[2]});
    else
        rtn = compressor.copy_data(reinterpret_cast<const float*>(data), conf.num,
                                {conf.dims[1], conf.dims[0], 1}, {chunks[0], chunks[1], chunks[2]});//temp 2D support. not sure if works well.
    compressor.set_target_pwe(conf.absErrorBound);
    rtn = compressor.compress();
    auto stream = compressor.get_encoded_bitstream();
        
    char * outData=new char[stream.size()+conf.size_est()];
    outSize=stream.size();
    memcpy(outData,stream.data(),stream.size());//maybe not efficient
    stream.clear();
    stream.shrink_to_fit();
    return outData;

}
template<class T, QoZ::uint N> 
void SPERR_Decompress(char *cmpData, size_t cmpSize, T *decData){
    
    std::vector<uint8_t> in_stream(cmpData,cmpData+cmpSize);
    SPERR3D_OMP_D decompressor;
  
    decompressor.set_num_threads(1);
    if (decompressor.use_bitstream(in_stream.data(), in_stream.size()) != sperr::RTNType::Good) {
        std::cerr << "Read compressed file error: "<< std::endl;
        return;
    }

    if (decompressor.decompress(in_stream.data()) != sperr::RTNType::Good) {
        std::cerr << "Decompression failed!" << std::endl;
        return ;
    }
   
    in_stream.clear();
    in_stream.shrink_to_fit();
    const auto vol = decompressor.get_data<float>();
    memcpy(decData,vol.data(),sizeof(T)*vol.size());//maybe not efficient
    return;
}




template<class T, QoZ::uint N>
char * outlier_compress(QoZ::Config &conf,T *data,size_t &outSize){

    char * outlier_compress_output;
    if (conf.offsetPredictor ==0){
        auto quantizer = QoZ::LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto sz = QoZ::make_sz_general_compressor<T, 1>(QoZ::make_sz_general_frontend<T, 1>(conf, QoZ::ZeroPredictor<T, 1>(), quantizer), QoZ::HuffmanEncoder<int>(),
                                                                       QoZ::Lossless_zstd());  
        outlier_compress_output =  (char *)sz->compress(conf,data,outSize);
        delete sz;
    }
    else if (conf.offsetPredictor ==1){
        conf.lorenzo = true;
        conf.lorenzo2 = true;
        conf.regression = false;
        conf.regression2 = false;
        conf.openmp = false;
        conf.blockSize = 16;//original 5
        conf.quantbinCnt = 65536 * 2;

        auto quantizer = QoZ::LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto sz = make_lorenzo_regression_compressor<T, 1>(conf, quantizer, QoZ::HuffmanEncoder<int>(), QoZ::Lossless_zstd());
        outlier_compress_output =  (char *)sz->compress(conf,data,outSize);
        delete sz;
    }
    else if (conf.offsetPredictor == 2){
        conf.setDims(conf.dims.begin(),conf.dims.end());
        conf.lorenzo = true;
        conf.lorenzo2 = true;
        conf.regression = false;
        conf.regression2 = false;
        conf.openmp = false;
        conf.blockSize = 5;
        conf.quantbinCnt = 65536 * 2;
        auto quantizer = QoZ::LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2);
        auto sz = make_lorenzo_regression_compressor<T, N>(conf, quantizer, QoZ::HuffmanEncoder<int>(), QoZ::Lossless_zstd());
        outlier_compress_output =  (char *)sz->compress(conf,data,outSize);
        delete sz;
    }

    else if (conf.offsetPredictor == 3){
        conf.interpAlgo=QoZ::INTERP_ALGO_CUBIC;
        conf.interpDirection=0;
        auto sz = QoZ::SZInterpolationCompressor<T, 1, QoZ::LinearQuantizer<T>, QoZ::HuffmanEncoder<int>, QoZ::Lossless_zstd>(
        QoZ::LinearQuantizer<T>(conf.absErrorBound),
        QoZ::HuffmanEncoder<int>(),
        QoZ::Lossless_zstd());
        outlier_compress_output =  (char *)sz.compress(conf,data,outSize);
        
    }

    else if (conf.offsetPredictor == 4){
            
        conf.setDims(conf.dims.begin(),conf.dims.end());
        conf.interpAlgo=QoZ::INTERP_ALGO_CUBIC;
        conf.interpDirection=0;
        auto sz = QoZ::SZInterpolationCompressor<T, N, QoZ::LinearQuantizer<T>, QoZ::HuffmanEncoder<int>, QoZ::Lossless_zstd>(
        QoZ::LinearQuantizer<T>(conf.absErrorBound),
        QoZ::HuffmanEncoder<int>(),
        QoZ::Lossless_zstd());
            
       
        outlier_compress_output =  (char *)sz.compress(conf,data,outSize);
        
    }
    return outlier_compress_output;

}

template<class T, QoZ::uint N>
void outlier_decompress(QoZ::Config &conf,char *cmprData,size_t outSize,T*decData){
    if (conf.offsetPredictor ==0){
        auto sz = QoZ::make_sz_general_compressor<T, 1>(QoZ::make_sz_general_frontend<T, 1>(conf, QoZ::ZeroPredictor<T, 1>(), QoZ::LinearQuantizer<T>()), QoZ::HuffmanEncoder<int>(),
                                                                       QoZ::Lossless_zstd());

        sz->decompress((QoZ::uchar *)cmprData,outSize,decData);
       
        delete sz;
    }

    else if (conf.offsetPredictor ==1){
        conf.lorenzo = true;
        conf.lorenzo2 = true;
        conf.regression = false;
        conf.regression2 = false;
        conf.openmp = false;
        conf.blockSize = 16;//original 5
        conf.quantbinCnt = 65536 * 2;

        auto sz = make_lorenzo_regression_compressor<T, 1>(conf, QoZ::LinearQuantizer<T>(), QoZ::HuffmanEncoder<int>(), QoZ::Lossless_zstd());
                  
        sz->decompress((QoZ::uchar *)cmprData,outSize,decData);
        delete sz;
    }
    else if (conf.offsetPredictor == 2){
        conf.setDims(conf.dims.begin(),conf.dims.end());
        conf.lorenzo = true;
        conf.lorenzo2 = true;
        conf.regression = false;
        conf.regression2 = false;
        conf.openmp = false;
        conf.blockSize = 5;
        conf.quantbinCnt = 65536 * 2;

        auto sz = make_lorenzo_regression_compressor<T, N>(conf, QoZ::LinearQuantizer<T>(), QoZ::HuffmanEncoder<int>(), QoZ::Lossless_zstd());       
        sz->decompress((QoZ::uchar *)cmprData,outSize,decData);
        delete sz;
    }

    else if (conf.offsetPredictor == 3){
        conf.interpAlgo=QoZ::INTERP_ALGO_CUBIC;
        conf.interpDirection=0;

           
        auto sz = QoZ::SZInterpolationCompressor<T, 1, QoZ::LinearQuantizer<T>, QoZ::HuffmanEncoder<int>, QoZ::Lossless_zstd>(
        QoZ::LinearQuantizer<T>(),
        QoZ::HuffmanEncoder<int>(),
        QoZ::Lossless_zstd());      
        sz.decompress((QoZ::uchar *)cmprData,outSize,decData);
        
    }

    else if (conf.offsetPredictor == 4){
            
        conf.setDims(conf.dims.begin(),conf.dims.end());
        conf.interpAlgo=QoZ::INTERP_ALGO_CUBIC;
        conf.interpDirection=0;          
        auto sz = QoZ::SZInterpolationCompressor<T, N, QoZ::LinearQuantizer<T>, QoZ::HuffmanEncoder<int>, QoZ::Lossless_zstd>(
        QoZ::LinearQuantizer<T>(),
        QoZ::HuffmanEncoder<int>(),
        QoZ::Lossless_zstd());      
        sz.decompress((QoZ::uchar *)cmprData,outSize,decData);
        
    }
    
}

template<class T, QoZ::uint N>
char *SZ_compress_Interp(QoZ::Config &conf, T *data, size_t &outSize) {

//    std::cout << "****************** Interp Compression ****************" << std::endl;
//    std::cout << "Interp Op          = " << interpAlgo << std::endl
//              << "Direction          = " << direction << std::endl
//              << "SZ block size      = " << blockSize << std::endl
//              << "Interp block size  = " << interpBlockSize << std::endl;

    assert(N == conf.N);
    assert(conf.cmprAlgo == QoZ::ALGO_INTERP);
    QoZ::calAbsErrorBound(conf, data);

    //conf.print();
    
    auto sz = QoZ::SZInterpolationCompressor<T, N, QoZ::LinearQuantizer<T>, QoZ::HuffmanEncoder<int>, QoZ::Lossless_zstd>(
            QoZ::LinearQuantizer<T>(conf.absErrorBound),
            QoZ::HuffmanEncoder<int>(),
            QoZ::Lossless_zstd());

   
    //QoZ::Timer timer;

    //timer.start();
    char *cmpData = (char *) sz.compress(conf, data, outSize);
     //double incall_time = timer.stop();
    //std::cout << "incall time = " << incall_time << "s" << std::endl;
    return cmpData;
}

template<class T, QoZ::uint N>
void SZ_decompress_Interp(QoZ::Config &conf, char *cmpData, size_t cmpSize, T *decData) {
    /*

    assert(conf.cmprAlgo == QoZ::ALGO_INTERP);
    QoZ::uchar const *cmpDataPos = (QoZ::uchar *) cmpData;
    if (conf.wavelet==0 and !use_sperr<T,N>(conf)){
        
        auto sz = QoZ::SZInterpolationCompressor<T, N, QoZ::LinearQuantizer<T>, QoZ::HuffmanEncoder<int>, QoZ::Lossless_zstd>(
                QoZ::LinearQuantizer<T>(),
                QoZ::HuffmanEncoder<int>(),
                QoZ::Lossless_zstd());
        if (!conf.blockwiseTuning)
            sz.decompress(cmpDataPos, cmpSize, decData);
        else{
            sz.decompress_block(cmpDataPos, cmpSize, decData);
        }
    }
    
    else{


        if(use_sperr<T,N>(conf) and conf.wavelet<=1){
            std::vector<uint8_t> in_stream(cmpData,cmpData+cmpSize);
            SPERR3D_OMP_D decompressor;
            decompressor.set_num_threads(1);
            if (decompressor.use_bitstream(in_stream.data(), in_stream.size()) != sperr::RTNType::Good) {
                std::cerr << "Read compressed file error: "<< std::endl;
                return;
            }

            if (decompressor.decompress(in_stream.data()) != sperr::RTNType::Good) {
                std::cerr << "Decompression failed!" << std::endl;
                return ;
            }
            in_stream.clear();
            in_stream.shrink_to_fit();
            const auto vol = decompressor.get_data<float>();
            memcpy(decData,vol.data(),sizeof(T)*conf.num);//maybe not efficient
            
            return;




        }
      
        size_t first =conf.firstSize;
        size_t second=cmpSize-conf.firstSize;    

        if(use_sperr<T,N>(conf))
            SPERR_Decompress<T,N>((char*)cmpDataPos, first,decData);
        else{
            auto sz = QoZ::SZInterpolationCompressor<T, N, QoZ::LinearQuantizer<T>, QoZ::HuffmanEncoder<int>, QoZ::Lossless_zstd>(
                    QoZ::LinearQuantizer<T>(),
                    QoZ::HuffmanEncoder<int>(),
                    QoZ::Lossless_zstd());

        
            if (!conf.blockwiseTuning)
                sz.decompress(cmpDataPos, first, decData);
            else{
               
                sz.decompress_block(cmpDataPos, first, decData);
            }
        }
      
        //QoZ::writefile<T>("waved.qoz.dec.sigmo", decData, conf.num);
     

         //QoZ::writefile<T>("waved.qoz.dec.logit", decData, conf.num);

        if(conf.wavelet>1){
            T* newDecData;
            if(conf.pyBind){
              
                
                std::vector<size_t> ori_dims=conf.dims;
                size_t ori_num=conf.num;
                conf.dims=conf.coeffs_dims;
                conf.num=conf.coeffs_num; 
                newDecData= QoZ::pybind_wavelet_postprocessing<T,N>(conf,decData,conf.metadata,conf.wavelet, false,ori_dims);
                conf.dims=ori_dims;
                conf.num=ori_num;
               
                
                
            }
            else
                newDecData= QoZ::external_wavelet_postprocessing<T,N>(decData, conf.coeffs_dims, conf.coeffs_num, conf.wavelet, conf.pid, false,conf.dims);

            
          
            delete []decData;
            decData = new T [conf.num];
            memcpy(decData,newDecData,sizeof(T)*conf.num);//maybe not efficient
            delete []newDecData;

        
        }
        
        else{
            QoZ::Wavelet<T,N> wlt;
            wlt.postProcess_cdf97(decData,conf.dims);
        }
       
        if(conf.conditioning and (!use_sperr<T,N>(conf) or conf.wavelet>1)){
            auto rtn=post_Condition<T,N>(decData,conf.num,conf.meta);
                
        }

       
        //QoZ::writefile<T>("waved.qoz.dec.idwt", decData, conf.num);
       
        
        if(second>0){
            T *offsets =new T [conf.num];
            outlier_decompress<T,N>(conf,(char*)(cmpDataPos+first),second,offsets);
        
        
        //QoZ::writefile<T>("waved.qoz.dec.offset", offsets, conf.num); 
            for(size_t i=0;i<conf.num;i++)
                decData[i]+=offsets[i];//maybe not efficient
            delete [] offsets;
        }
        
    }    
    */
}


template<class T, QoZ::uint N>
double do_not_use_this_interp_compress_block_test(T *data, std::vector<size_t> dims, size_t num,
                                                  double eb, int interp_op, int direction_op, int block_size) {
    std::vector<T> data1(data, data + num);
    size_t outSize = 0;
    QoZ::Config conf;
    conf.absErrorBound = eb;
    conf.setDims(dims.begin(), dims.end());
    conf.blockSize = block_size;
    conf.interpAlgo = interp_op;
    conf.interpDirection = direction_op;
    auto sz = QoZ::SZBlockInterpolationCompressor<T, N, QoZ::LinearQuantizer<T>, QoZ::HuffmanEncoder<int>, QoZ::Lossless_zstd>(
            QoZ::LinearQuantizer<T>(eb),
            QoZ::HuffmanEncoder<int>(),
            QoZ::Lossless_zstd());
    char *cmpData = (char *) sz.compress(conf, data1.data(), outSize);
    delete[]cmpData;
    auto compression_ratio = num * sizeof(T) * 1.0 / outSize;
    return compression_ratio;
}



/*
template<class T, QoZ::uint N>
int compareWavelets(QoZ::Config &conf, std::vector< std::vector<T> > & sampled_blocks){//This is an unfinished API. Not sure whether useful later.
    size_t sampleBlockSize=conf.sampleBlockSize;
    std::vector<size_t> global_dims=conf.dims;
    size_t global_num=conf.num;

    num_sampled_blocks=sampled_blocks.size();
    per_block_ele_num=pow(sampleBlockSize+1,N);
    ele_num=num_sampled_blocks*per_block_ele_num;
    conf.dims=std::vector<size_t>(N,sampleBlockSize+1);
    conf.num=per_block_ele_num;
    std::vector<T> cur_block(per_block_ele_num,0);

    double wave_eb=conf.absErrorBound*conf.wavelet_rel_coeff;

    size_t sig_count=0;

    std::vector<T> gathered_coeffs;
    std::vector<T> gathered_blocks;

    return 0;

}
*/


template<class T, QoZ::uint N>
void sampleBlocks(T *data,std::vector<size_t> &dims, size_t sampleBlockSize,std::vector< std::vector<T> > & sampled_blocks,double sample_rate,int profiling ,std::vector<std::vector<size_t> > &starts,int var_first=0){
    for(int i=0;i<sampled_blocks.size();i++){
                std::vector< T >().swap(sampled_blocks[i]);                
            }
            std::vector< std::vector<T> >().swap(sampled_blocks);
    for(int i=0;i<sampled_blocks.size();i++){
        std::vector< T >().swap(sampled_blocks[i]);                  
    }
    std::vector< std::vector<T> >().swap(sampled_blocks);                               
    size_t totalblock_num=1;
    for(int i=0;i<N;i++){                        
        totalblock_num*=(int)((dims[i]-1)/sampleBlockSize);
    }               
    size_t idx=0,block_idx=0;   
    if(profiling){
        size_t num_filtered_blocks=starts.size();    
        if(var_first==0){  
            size_t sample_stride=(size_t)(num_filtered_blocks/(totalblock_num*sample_rate));
            if(sample_stride<=0)
                sample_stride=1;
            
            for(size_t i=0;i<num_filtered_blocks;i+=sample_stride){
                std::vector<T> s_block;
                QoZ::sample_blocks<T,N>(data, s_block,dims, starts[i],sampleBlockSize+1);
                sampled_blocks.push_back(s_block);
                
            }
            
        }
        else{
            std::vector< std::pair<double,std::vector<size_t> > >block_heap;
            for(size_t i=0;i<num_filtered_blocks;i++){
                double mean,sigma2,range;
                QoZ::blockwise_profiling<T>(data,dims, starts[i],sampleBlockSize+1, mean,sigma2,range);
                block_heap.push_back(std::pair<double,std::vector<size_t> >(sigma2,starts[i]));
                
            }
            std::make_heap(block_heap.begin(),block_heap.end());
          

            size_t sampled_block_num=totalblock_num*sample_rate;
            if(sampled_block_num>num_filtered_blocks)
                sampled_block_num=num_filtered_blocks;
            if(sampled_block_num==0)
                sampled_block_num=1;

            for(size_t i=0;i<sampled_block_num;i++){
                std::vector<T> s_block;
             
                QoZ::sample_blocks<T,N>(data, s_block,dims, block_heap.front().second,sampleBlockSize+1);
              
                sampled_blocks.push_back(s_block);
                std::pop_heap(block_heap.begin(),block_heap.end());
                block_heap.pop_back();
               
            }
        }
    }               
    else{
        if(var_first==0){
            size_t sample_stride=(size_t)(1.0/sample_rate);
            if(sample_stride<=0)
                sample_stride=1;
            if (N==2){                        
                for (size_t x_start=0;x_start<dims[0]-sampleBlockSize;x_start+=sampleBlockSize){                           
                    for (size_t y_start=0;y_start<dims[1]-sampleBlockSize;y_start+=sampleBlockSize){
                        if (idx%sample_stride==0){
                            std::vector<size_t> starts{x_start,y_start};
                            std::vector<T> s_block;
                            QoZ::sample_blocks<T,N>(data, s_block,dims, starts,sampleBlockSize+1);
                            sampled_blocks.push_back(s_block);
                        }
                        idx+=1;
                    }
                }
            }
            else if (N==3){                  
                for (size_t x_start=0;x_start<dims[0]-sampleBlockSize;x_start+=sampleBlockSize){                          
                    for (size_t y_start=0;y_start<dims[1]-sampleBlockSize;y_start+=sampleBlockSize){
                        for (size_t z_start=0;z_start<dims[2]-sampleBlockSize;z_start+=sampleBlockSize){
                            if (idx%sample_stride==0){
                                std::vector<size_t> starts{x_start,y_start,z_start};
                                std::vector<T> s_block;
                                QoZ::sample_blocks<T,N>(data, s_block,dims, starts,sampleBlockSize+1);
                                sampled_blocks.push_back(s_block);
                            }
                            idx+=1;
                        }
                    }
                }
            }
        }
        else{
            std::vector <std::vector<size_t> > blocks_starts;
            if (N==2){  
                for (size_t x_start=0;x_start<dims[0]-sampleBlockSize;x_start+=sampleBlockSize){                           
                    for (size_t y_start=0;y_start<dims[1]-sampleBlockSize;y_start+=sampleBlockSize){
                       
                            blocks_starts.push_back(std::vector<size_t>{x_start,y_start});
                    }
                }

            }
            else if (N==3){           
                for (size_t x_start=0;x_start<dims[0]-sampleBlockSize;x_start+=sampleBlockSize){                          
                    for (size_t y_start=0;y_start<dims[1]-sampleBlockSize;y_start+=sampleBlockSize){
                        for (size_t z_start=0;z_start<dims[2]-sampleBlockSize;z_start+=sampleBlockSize){
                            blocks_starts.push_back(std::vector<size_t>{x_start,y_start,z_start});
                        }
                    }
                }
            

                std::vector< std::pair<double,std::vector<size_t> > >block_heap;
                for(size_t i=0;i<totalblock_num;i++){
                    double mean,sigma2,range;
                    QoZ::blockwise_profiling<T>(data,dims, blocks_starts[i],sampleBlockSize+1, mean,sigma2,range);
                    block_heap.push_back(std::pair<double,std::vector<size_t> >(sigma2,blocks_starts[i]));
                }
                std::make_heap(block_heap.begin(),block_heap.end());
                size_t sampled_block_num=totalblock_num*sample_rate;
                if(sampled_block_num==0)
                    sampled_block_num=1;
                for(size_t i=0;i<sampled_block_num;i++){
                    std::vector<T> s_block;
                    QoZ::sample_blocks<T,N>(data, s_block,dims, block_heap.front().second,sampleBlockSize+1);
                    sampled_blocks.push_back(s_block);
                    std::pop_heap(block_heap.begin(),block_heap.end());
                    block_heap.pop_back();
                }

            }
        }
    }
}


template<class T, QoZ::uint N>
std::pair<double,double> CompressTest(const QoZ::Config &conf,const std::vector< std::vector<T> > & sampled_blocks,QoZ::ALGO algo = QoZ::ALGO_INTERP,
                    QoZ::TUNING_TARGET tuningTarget=QoZ::TUNING_TARGET_RD,bool useFast=true,double profiling_coeff=1,const std::vector<double> &orig_means=std::vector<double>(),
                    const std::vector<double> &orig_sigma2s=std::vector<double>(),const std::vector<double> &orig_ranges=std::vector<double>(),const std::vector<T> &flattened_sampled_data=std::vector<T>(),const std::vector< std::vector<T> > & waveleted_input=std::vector< std::vector<T> >()){
    QoZ::Config testConfig(conf);
    size_t ssim_size=conf.SSIMBlockSize;    
    if(algo == QoZ::ALGO_LORENZO_REG){
        testConfig.cmprAlgo = QoZ::ALGO_LORENZO_REG;
        testConfig.dims=conf.dims;
        testConfig.num=conf.num;
        testConfig.lorenzo = true;
        testConfig.lorenzo2 = true;
        testConfig.regression = false;
        testConfig.regression2 = false;
        testConfig.openmp = false;
        testConfig.blockSize = 5;//why?
        testConfig.quantbinCnt = 65536 * 2;
    }
    double square_error=0.0;
    double bitrate=0.0;
    double metric=0.0;
    size_t sampleBlockSize=testConfig.sampleBlockSize;
    size_t num_sampled_blocks=sampled_blocks.size();
    size_t per_block_ele_num=pow(sampleBlockSize+1,N);
    size_t ele_num=num_sampled_blocks*per_block_ele_num;
    std::vector<T> cur_block(testConfig.num,0);
    std::vector<int> q_bins;
    std::vector<std::vector<int> > block_q_bins;
    std::vector<size_t> q_bin_counts;
    std::vector<T> flattened_cur_blocks;
    size_t idx=0;   
    QoZ::concepts::CompressorInterface<T> *sz;
    size_t totalOutSize=0;
    if(algo == QoZ::ALGO_LORENZO_REG){
        auto quantizer = QoZ::LinearQuantizer<T>(testConfig.absErrorBound, testConfig.quantbinCnt / 2);
        if (useFast &&N == 3 && !testConfig.regression2) {
            sz = QoZ::make_sz_general_compressor<T, N>(QoZ::make_sz_fast_frontend<T, N>(testConfig, quantizer), QoZ::HuffmanEncoder<int>(),
                                                                   QoZ::Lossless_zstd());
        }
        else{
            sz = make_lorenzo_regression_compressor<T, N>(testConfig, quantizer, QoZ::HuffmanEncoder<int>(), QoZ::Lossless_zstd());

        }
    }
    else if(algo == QoZ::ALGO_INTERP){

        sz =  new QoZ::SZInterpolationCompressor<T, N, QoZ::LinearQuantizer<T>, QoZ::HuffmanEncoder<int>, QoZ::Lossless_zstd>(
                        QoZ::LinearQuantizer<T>(testConfig.absErrorBound),
                        QoZ::HuffmanEncoder<int>(),
                        QoZ::Lossless_zstd());

    }
    else{
        std::cout<<"algo type error!"<<std::endl;
        return std::pair<double,double>(0,0);
    }
                           
    for (int k=0;k<num_sampled_blocks;k++){
        size_t sampleOutSize;
        std::vector<T> cur_block(testConfig.num);
        if(testConfig.wavelet==0 or waveleted_input.size()==0){
            std::copy(sampled_blocks[k].begin(),sampled_blocks[k].end(),cur_block.begin());
        }
        else{
            std::copy(waveleted_input[k].begin(),waveleted_input[k].end(),cur_block.begin());

        }
        char *cmprData;
        if(use_sperr<T,N>(testConfig)){
            if(testConfig.wavelet<=1){
                cmprData=SPERR_Compress<T,N>(testConfig,cur_block.data(),sampleOutSize);
                totalOutSize+=sampleOutSize;
                if(tuningTarget!=QoZ::TUNING_TARGET_CR){
                    SPERR_Decompress<T,N>(cmprData,sampleOutSize,cur_block.data());
                    
                } 
            }
            else{
    
                cmprData=SPERR_Compress<T,N>(testConfig,cur_block.data(),sampleOutSize);
                
                totalOutSize+=sampleOutSize;
                if(1){//tuningTarget!=QoZ::TUNING_TARGET_CR){
                    SPERR_Decompress<T,N>(cmprData,sampleOutSize,cur_block.data());
                    std::vector<size_t> ori_sbs(N,testConfig.sampleBlockSize+1);
                    T *idwtData;
                   // if(conf.pyBind)
                       // idwtData=QoZ::pybind_wavelet_postprocessing<T,N>(testConfig,cur_block.data(), testConfig.metadata,testConfig.wavelet, false,ori_sbs);
                    //else

                        idwtData=QoZ::external_wavelet_postprocessing<T,N>(cur_block.data(),testConfig.dims, testConfig.num, testConfig.wavelet, testConfig.pid, false,ori_sbs);
                    

                    cur_block.assign(idwtData,idwtData+per_block_ele_num);//maybe not efficient, what about change the return meta of ewp?
                    delete []idwtData;
                    if(testConfig.conditioning){
                        post_Condition<T,N>(cur_block.data(),per_block_ele_num,testConfig.block_metas[k]);
                    }
                    
                    std::vector<T> offsets(per_block_ele_num);
                    
                    for(size_t i=0;i<per_block_ele_num;i++)
                        offsets[i]=sampled_blocks[k][i]-cur_block[i];
                    
                    size_t oc_size;
                    std::vector<size_t> ori_dims=testConfig.dims,temp_dims={per_block_ele_num};

                    testConfig.setDims(temp_dims.begin(),temp_dims.end());

                    char * offsetsCmprData=outlier_compress<T,N>(testConfig,offsets.data(),oc_size);
                    testConfig.setDims(ori_dims.begin(),ori_dims.end());
                    delete []offsetsCmprData;
                    totalOutSize+=oc_size;
                    for(size_t i=0;i<per_block_ele_num;i++)
                        cur_block[i]+=offsets[i];

                }
                



            }
   
            delete []cmprData;          
        }    
        else{
            cmprData = (char*)sz->compress(testConfig, cur_block.data(), sampleOutSize,1);

            delete[]cmprData;
            if(testConfig.wavelet>0 and waveleted_input.size()>0 and tuningTarget!=QoZ::TUNING_TARGET_CR){
                
                if(testConfig.wavelet==1){
                    QoZ::Wavelet<T,N> wlt;
                    wlt.postProcess_cdf97(cur_block.data(),conf.dims);
                    
                }
                else{
                    std::vector<size_t> ori_sbs(N,testConfig.sampleBlockSize+1);
                    T *idwtData;
                    
                        idwtData=QoZ::external_wavelet_postprocessing<T,N>(cur_block.data(),testConfig.dims, testConfig.num, testConfig.wavelet, testConfig.pid, false,ori_sbs);
            
                    cur_block.assign(idwtData,idwtData+per_block_ele_num);//maybe not efficient, what about change the return type of ewp?
                    delete []idwtData;
                }
                if(testConfig.conditioning){
                    
                    post_Condition<T,N>(cur_block.data(),per_block_ele_num,testConfig.block_metas[k]);
                }

            }
        }

        
        if(algo==QoZ::ALGO_INTERP and !(use_sperr<T,N>(testConfig)))
            block_q_bins.push_back(testConfig.quant_bins);

        if(tuningTarget==QoZ::TUNING_TARGET_RD){
            if(algo==QoZ::ALGO_INTERP and !(use_sperr<T,N>(testConfig)) )
                square_error+=testConfig.decomp_square_error;
            else{
               
                for(size_t j=0;j<per_block_ele_num;j++){
                    T value=sampled_blocks[k][j]-cur_block[j];
                    square_error+=value*value;
                
                }
            }
        }
        else if (tuningTarget==QoZ::TUNING_TARGET_SSIM){
            size_t ssim_block_num=orig_means.size();                       
            double mean=0,sigma2=0,cov=0,range=0;
            double orig_mean=0,orig_sigma2=0,orig_range=0;  
            std::vector<size_t>block_dims(N,sampleBlockSize+1);                      
            if(N==2){
                for (size_t i=0;i+ssim_size<sampleBlockSize+1;i+=ssim_size){
                    for (size_t j=0;j+ssim_size<sampleBlockSize+1;j+=ssim_size){
                        orig_mean=orig_means[idx];
                        orig_sigma2=orig_sigma2s[idx];
                        orig_range=orig_ranges[idx];
                        std::vector<size_t> starts{i,j};
                        QoZ::blockwise_profiling<T>(cur_block.data(),block_dims,starts,ssim_size,mean,sigma2,range);
                        cov=QoZ::blockwise_cov<T>(sampled_blocks[k].data(),cur_block.data(),block_dims,starts,ssim_size,orig_mean,mean);
                        metric+=QoZ::SSIM(orig_range,orig_mean,orig_sigma2,mean,sigma2,cov)/ssim_block_num;
                        idx++;


                    }
                }
            }
            else if(N==3){
                for (size_t i=0;i+ssim_size<sampleBlockSize+1;i+=ssim_size){
                    for (size_t j=0;j+ssim_size<sampleBlockSize+1;j+=ssim_size){
                        for (size_t kk=0;kk+ssim_size<sampleBlockSize+1;kk+=ssim_size){
                            orig_mean=orig_means[idx];
                            orig_sigma2=orig_sigma2s[idx];
                            orig_range=orig_ranges[idx];
                            std::vector<size_t> starts{i,j,kk};
                            QoZ::blockwise_profiling<T>(cur_block.data(),block_dims,starts,ssim_size,mean,sigma2,range);
                            cov=QoZ::blockwise_cov<T>(sampled_blocks[k].data(),cur_block.data(),block_dims,starts,ssim_size,orig_mean,mean);
                            //printf("%.8f %.8f %.8f %.8f %.8f %.8f %.8f\n",orig_range,orig_sigma2,orig_mean,range,sigma2,mean,cov);
                            metric+=QoZ::SSIM(orig_range,orig_mean,orig_sigma2,mean,sigma2,cov)/ssim_block_num;
                     
                            idx++;
                        }
                    }
                }
            }
        }
        else if (tuningTarget==QoZ::TUNING_TARGET_AC){
            flattened_cur_blocks.insert(flattened_cur_blocks.end(),cur_block.begin(),cur_block.end());
        }                      
    }
    if(algo==QoZ::ALGO_INTERP and !(use_sperr<T,N>(testConfig))){
        q_bin_counts=testConfig.quant_bin_counts;
        size_t level_num=q_bin_counts.size();
        size_t last_pos=0;
        for(int k=level_num-1;k>=0;k--){
            for (size_t l =0;l<num_sampled_blocks;l++){
                for (size_t m=last_pos;m<q_bin_counts[k];m++){
                    q_bins.push_back(block_q_bins[l][m]);
                }
            }
            last_pos=q_bin_counts[k];
        }      
    }
    size_t sampleOutSize;
    if(!use_sperr<T,N>(testConfig)){
        auto cmprData=sz->encoding_lossless(totalOutSize,q_bins);                   
        delete[]cmprData;
       
    }    
   
    bitrate=8*double(totalOutSize)/ele_num;
    
    bitrate*=profiling_coeff;
    if(tuningTarget==QoZ::TUNING_TARGET_RD){
                   
        double mse=square_error/ele_num;
        mse*=profiling_coeff;      
        if(testConfig.wavelet==1)
            mse*=testConfig.waveletMseFix;
        else if(testConfig.wavelet>1)
            mse*=testConfig.waveletMseFix2;
        metric=QoZ::PSNR(testConfig.rng,mse);
    }
    else if (tuningTarget==QoZ::TUNING_TARGET_AC){                       
        metric=1.0-QoZ::autocorrelation<T>(flattened_sampled_data.data(),flattened_cur_blocks.data(),ele_num);                        
    }                    
    //printf("%.2f %.2f %.4f %.2f\n",testConfig.alpha,testConfig.beta,bitrate,metric);   
    if(testConfig.wavelet==1){
        bitrate*=testConfig.waveletBrFix;
    } 
    else if(testConfig.wavelet>1){
        bitrate*=testConfig.waveletBrFix2;
    }       

    if(algo==QoZ::ALGO_LORENZO_REG)    {
        bitrate*=testConfig.lorenzoBrFix;
    }
    delete sz;
    return std::pair(bitrate,metric);
}



template<class T, QoZ::uint N> 
double estimateSPERRCRbasedonErrorBound(double error_bound,T * data, double sample_rate, size_t blocksize,std::vector<size_t> &dims,int profiling=0,int var_first=0){

    std::vector< std::vector<T> > sampled_blocks;
   
    //size_t num_sampled_blocks;
    //size_t per_block_ele_num;
    //size_t ele_num;

    QoZ::Config conf(1);//maybe a better way exist?
    conf.setDims(dims.begin(),dims.end());
    conf.sperr=1;
    conf.wavelet=1;
    conf.wavelet_rel_coeff=1.5;
    conf.profiling=profiling;
    conf.var_first=var_first;
    conf.sampleBlockSize=blocksize;
    conf.cmprAlgo=QoZ::ALGO_INTERP;
    conf.tuningTarget=QoZ::TUNING_TARGET_CR;
    conf.errorBoundMode=QoZ::EB_ABS;
    conf.absErrorBound=error_bound;
    size_t totalblock_num=1;  
    for(int i=0;i<N;i++){                      
        totalblock_num*=(size_t)((conf.dims[i]-1)/conf.sampleBlockSize);
    }
    std::cout<<"t1"<<std::endl;

    std::vector<std::vector<size_t> >starts;
    if((conf.waveletTuningRate>0 or conf.autoTuningRate>0 or conf.predictorTuningRate>0) and conf.profiling){      
        conf.profStride=conf.sampleBlockSize/4;
        if(N==2){
            QoZ::profiling_block_2d<T,N>(data,conf.dims,starts,blocksize,conf.absErrorBound,conf.profStride);
        }
        else if (N==3){
            QoZ::profiling_block_3d<T,N>(data,conf.dims,starts,blocksize,conf.absErrorBound,conf.profStride);
        }
       
    }
    std::cout<<"t2"<<std::endl;

    size_t num_filtered_blocks=starts.size();

    sampleBlocks<T,N>(data,conf.dims,conf.sampleBlockSize,sampled_blocks,sample_rate,conf.profiling,starts,conf.var_first);
    std::cout<<"t3"<<std::endl;
           
    //num_sampled_blocks=sampled_blocks.size();
    //per_block_ele_num=pow(sampleBlockSize+1,N);
   // ele_num=num_sampled_blocks*per_block_ele_num;

    std::pair<double,double> results=CompressTest<T,N>(conf, sampled_blocks,QoZ::ALGO_INTERP,QoZ::TUNING_TARGET_CR,false);

    double cur_ratio=sizeof(T)*8.0/results.first;
    std::cout<<"t4 "<<cur_ratio<<std::endl;

    return cur_ratio;


    /*
    conf.dims=std::vector<size_t>(N,sampleBlockSize+1);
    conf.num=per_block_ele_num;
    std::vector<T> cur_block(per_block_ele_num,0);
    */

}




template<class T, QoZ::uint N>
char *SZ_compress_Interp_lorenzo(QoZ::Config &conf, T *data, size_t &outSize) {
    assert(conf.cmprAlgo == QoZ::ALGO_INTERP_LORENZO);
    QoZ::calAbsErrorBound(conf, data);

    double sample_rate=0.01;
    size_t blocksize=32;
    std::cout<<"estimated cr:"<<estimateSPERRCRbasedonErrorBound<T,N>(conf.absErrorBound,data,sample_rate,blocksize,conf.dims);
    outSize=1;
    char * out=new char[1];
    out[0]='a';
    return out;

}



#endif