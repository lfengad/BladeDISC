
#include "tensorflow/compiler/mlir/xla/ral/context/tvm_kernel_collector.h"
#include "tensorflow/compiler/mlir/xla/ral/ral_logging.h"
#include <iostream>
#include <fstream>
#include "tensorflow/compiler/mlir/xla/ral/context/tvm_kernel_cache.h"

namespace tao {
namespace ral {
namespace tvm_impl  {

static KernelCollector kernel_collector_ins;

static bool GetCollectorEnable() {
   bool enable = false;
   const char* pro_str = getenv("DISC_KERNEL_PROFILING");
   if (pro_str) {
       enable = std::string(pro_str) == "1";
   } 
   return enable;
}

static std::string GetCacheLocation() {
    std::string path = ".";
    const char* pro_str = getenv("DISC_PROFILING_CACHE");
    if (pro_str) {
       path = std::string(pro_str);
    }     
    return path;
}

void CollectorCheckEnable() {
    kernel_collector_ins.CheckEnable();
}

template<typename InT, typename OutT, typename AlphaBeta> 
void CollectorAddGemmKernel(const std::string& device,
     int64_t m, int64_t n, int64_t k,
     bool trans_a, bool trans_b) {  
    kernel_collector_ins.AddGemmKernel<InT, OutT, AlphaBeta>(device, m, n, k, trans_a, trans_b);
}

void CollectorDumpResults() {
    kernel_collector_ins.DumpResults();
}

void CollectorAddKernel(const std::string& kernel_key) {
    kernel_collector_ins.AddKernel(kernel_key);
}

KernelCollector::KernelCollector() {
   CheckEnable();
}

void KernelCollector::CheckEnable() {
    std::lock_guard<std::mutex> lck(mtx_);
    auto pre_enable = enable_;
    enable_ = GetCollectorEnable();
    if (pre_enable && !enable_) {
        // VLOG(0) << "clear #####################";
        kernels_.clear();
    }
    if (pre_enable && !enable_) {
        // VLOG(0) << "reinit #####################";
        tvm_impl::TVMFuncCacheReInit();
    }
    TAO_VLOG(1) << "Collector check enable " << (enable_?1:0);
}

void KernelCollector::DumpResults() {
    std::lock_guard<std::mutex> lck(mtx_);
    if (!enable_) {
        return;
    }
    auto path = GetCacheLocation() + "/kernel_info.txt";
    std::ofstream file(path);
    std::ostringstream str;
    // VLOG(0) << "kernel size " << kernels_.size();
    for (auto it : kernels_) {
        str << it << std::endl;
    }
    // TAO_VLOG(0) << "Collected Kernel:\n" << str.str();
    file << str.str();
    file.close();
    TAO_VLOG(1) << "Collected Kernel to " << path << " with size " << kernels_.size();
}

KernelCollector::~KernelCollector() {
    DumpResults();
}

template<typename InT, typename OutT, typename AlphaBeta>
void KernelCollector::AddGemmKernel(const std::string& device,
     int64_t m, int64_t n, int64_t k,
     bool trans_a, bool trans_b) {
     if (enable_) {
        auto ta = trans_a ? tvm_impl::TVMGemmTranspose::Transpose
                    : tvm_impl::TVMGemmTranspose::NoTranspose;
        auto tb = trans_b ? tvm_impl::TVMGemmTranspose::Transpose
                                            : tvm_impl::TVMGemmTranspose::NoTranspose;
        auto key = tvm_impl::GetGemmTVMFuncKey<InT, OutT, AlphaBeta>(device, m, n, k, ta, tb);
        AddKernel(key);
     }
    //  VLOG(0) << "kernel name " << (uint64_t)this << " " << kernels_.size();
}


void KernelCollector::AddKernel(const std::string& kernel_key) {
    if (enable_) { 
        std::lock_guard<std::mutex> lck(mtx_);
        kernels_.emplace(kernel_key);
    }
}


template
void KernelCollector::AddGemmKernel<float, float, float>(const std::string& device,
     int64_t m, int64_t n, int64_t k,
     bool trans_a, bool trans_b);

template
void KernelCollector::AddGemmKernel<double, double, double>(const std::string& device,
     int64_t m, int64_t n, int64_t k,
     bool trans_a, bool trans_b);

template
void KernelCollector::AddGemmKernel<Eigen::half, Eigen::half, float>(const std::string& device,
     int64_t m, int64_t n, int64_t k,
     bool trans_a, bool trans_b);

template
void CollectorAddGemmKernel<float, float, float>(const std::string& device,
     int64_t m, int64_t n, int64_t k,
     bool trans_a, bool trans_b);

template
void CollectorAddGemmKernel<double, double, double>(const std::string& device,
     int64_t m, int64_t n, int64_t k,
     bool trans_a, bool trans_b);

template
void CollectorAddGemmKernel<Eigen::half, Eigen::half, float>(const std::string& device,
     int64_t m, int64_t n, int64_t k,
     bool trans_a, bool trans_b);

}
}
}