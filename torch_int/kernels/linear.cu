#include "include/linear.h"
#include "include/common.h"

#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>

// used by out_proj and fc2, return INT32
torch::Tensor linear_a8_w8_b32_o32(torch::Tensor input,  // INT8
                                   torch::Tensor weight, // INT8
                                   torch::Tensor bias    // INT32
) {
  auto M = input.size(0);
  auto N = weight.size(0);
  auto K = input.size(1);

  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = int32_t;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

#if CUDA_ARCH >= 800
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementComputeEpilogue,
          cutlass::epilogue::thread::ScaleType::NoBetaScaling>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
#elif CUDA_ARCH >= 750
  using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape,
      DefaultGemmCfg::InstructionShape,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementComputeEpilogue,
          cutlass::epilogue::thread::ScaleType::NoBetaScaling>>;
#elif CUDA_ARCH >= 700
  using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
      cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
      ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
      DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape,
      DefaultGemmCfg::InstructionShape,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 1, ElementAccumulator, ElementComputeEpilogue,
          cutlass::epilogue::thread::ScaleType::NoBetaScaling>>;
#else
  #error "Unsupported cuda arch"
#endif

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  auto device = input.device();
  // use the broadcasted bias as the output
  auto out = bias.to(device).view({1, -1}).repeat({M, 1});

  // constexpr int kSparse = Gemm::kSparse;
  // How many elements of A are covered per ElementE
  // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
  // The size of individual meta data
  // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      input.data_ptr<int8_t>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      weight.data_ptr<int8_t>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      out.data_ptr<int32_t>(), LayoutOutput::packed(output_size));

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);

  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,      // <- reference to matrix C on device
      out_ref,      // <- reference to matrix D on device
      {alpha},      1};
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot run");
  }

  return out;
}

// used by out_proj and fc2, return INT32
torch::Tensor linear_a8_w8_b32_o32_with_scaling(torch::Tensor input,  // INT8
                                                torch::Tensor weight, // INT8
                                                torch::Tensor bias,   // INT32
                                                float alpha,          // FP32
                                                float beta            // FP32
) {
  auto M = input.size(0);
  auto N = weight.size(0);
  auto K = input.size(1);

  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

#if CUDA_ARCH >= 800
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementComputeEpilogue>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
#elif CUDA_ARCH >= 750
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75>;
#elif CUDA_ARCH >= 700
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassSimt, cutlass::arch::Sm70>;
#else
  #error "Unsupported cuda arch"
#endif

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  auto device = input.device();
  // use the broadcasted bias as the output
  auto out = bias.to(device).view({1, -1}).repeat({M, 1});

  // constexpr int kSparse = Gemm::kSparse;
  // How many elements of A are covered per ElementE
  // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
  // The size of individual meta data
  // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      input.data_ptr<int8_t>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      weight.data_ptr<int8_t>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      out.data_ptr<int32_t>(), LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,      // <- reference to matrix C on device
      out_ref,      // <- reference to matrix D on device
      {alpha, beta}, 1};
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot run");
  }

  return out;
}

// used by out_proj and fc2, return FP32
torch::Tensor linear_a8_w8_bfp32_ofp32(torch::Tensor input,  // INT8
                                       torch::Tensor weight, // INT8
                                       torch::Tensor bias,   // FP32
                                       float alpha,          // FP32
                                       float beta            // FP32
) {
  auto M = input.size(0);
  auto N = weight.size(0);
  auto K = input.size(1);

  using ElementOutput = float;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

#if CUDA_ARCH >= 800
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementComputeEpilogue>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
#elif CUDA_ARCH >= 750
  using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape,
      DefaultGemmCfg::InstructionShape,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementComputeEpilogue>>;
#elif CUDA_ARCH >= 700
  using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
      cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
      ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
      DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape,
      DefaultGemmCfg::InstructionShape,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 1, ElementAccumulator, ElementComputeEpilogue>>;
#else
  #error "Unsupported cuda arch"
#endif

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  auto device = input.device();
  // use the broadcasted bias as the output
  auto out = bias.to(device).view({1, -1}).repeat({M, 1});

  // constexpr int kSparse = Gemm::kSparse;
  // How many elements of A are covered per ElementE
  // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
  // The size of individual meta data
  // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      input.data_ptr<ElementInputA>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      weight.data_ptr<ElementInputB>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      out.data_ptr<ElementOutput>(), LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,      // <- reference to matrix C on device
      out_ref,      // <- reference to matrix D on device
      {alpha, beta}, 1};
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot run");
  }

  return out;
}


// used by q_proj, k_proj, v_proj, return INT8
torch::Tensor linear_a8_w8_b8_o8(torch::Tensor input,  // INT8
                                 torch::Tensor weight, // INT8
                                 torch::Tensor bias,   // INT8
                                 float alpha,          // FP32
                                 float beta            // FP32
) {
  auto M = input.size(0);
  auto N = weight.size(0);
  auto K = input.size(1);

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

#if CUDA_ARCH >= 800
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::FastLinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
#elif CUDA_ARCH >= 750
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75>;
#elif CUDA_ARCH >= 700
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassSimt, cutlass::arch::Sm70>;
#else
  #error "Unsupported cuda arch"
#endif

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);
  auto device = input.device();
  auto out = bias.to(device).view({1, -1}).repeat({M, 1});

  // constexpr int kSparse = Gemm::kSparse;
  // How many elements of A are covered per ElementE
  // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
  // The size of individual meta data
  // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      input.data_ptr<int8_t>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      weight.data_ptr<int8_t>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      out.data_ptr<int8_t>(), LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,      // <- reference to matrix C on device
      out_ref,      // <- reference to matrix D on device
      {alpha, beta}, 1};
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement, status: " +
                             std::to_string((int)status));
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize, status: " +
                             std::to_string((int)status));
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot run, status: " +
                             std::to_string((int)status));
  }

  return out;
}

// used by fc1
torch::Tensor linear_relu_a8_w8_b8_o8(torch::Tensor input,  // INT8
                                      torch::Tensor weight, // INT8
                                      torch::Tensor bias,   // INT8
                                      float alpha,          // FP32
                                      float beta            // FP32
) {
  auto M = input.size(0);
  auto N = weight.size(0);
  auto K = input.size(1);

  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

#if CUDA_ARCH >= 800
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput, // <- data type of output matrix
      128 / cutlass::sizeof_bits<
                ElementOutput>::value, // <- this is the number of elements per
                                       // vectorized memory access. For half
                                       // precision, it's 8 elements. This
                                       // becomes the vector width of math
                                       // instructions in epilogue too
      ElementAccumulator,              // <- data type of accumulator
      ElementComputeEpilogue // <- data type for alpha in linear combination
                             // function
      >;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      EpilogueOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      3>;
#elif CUDA_ARCH >= 750
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput, // <- data type of output matrix
      128 / cutlass::sizeof_bits<
                ElementOutput>::value, // <- this is the number of elements per
                                       // vectorized memory access. For half
                                       // precision, it's 8 elements. This
                                       // becomes the vector width of math
                                       // instructions in epilogue too
      ElementAccumulator,              // <- data type of accumulator
      ElementComputeEpilogue // <- data type for alpha in linear combination
                             // function
      >;
  using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape,
      DefaultGemmCfg::InstructionShape,
      EpilogueOp>;
#elif CUDA_ARCH >= 700
  // LinearCombinationRelu does work with sm70, so we use torch relu instead.
  #define USE_TORCH_RELU
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput, 1, ElementAccumulator, ElementComputeEpilogue>;
  using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
      cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
      ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
      DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape,
      DefaultGemmCfg::InstructionShape,
      EpilogueOp>;
#else
  #error "Unsupported cuda arch"
#endif

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);
  auto device = input.device();
  // use the broadcasted bias as the output
  auto out = bias.to(device).view({1, -1}).repeat({M, 1});

  // constexpr int kSparse = Gemm::kSparse;
  // How many elements of A are covered per ElementE
  // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
  // The size of individual meta data
  // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      input.data_ptr<int8_t>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      weight.data_ptr<int8_t>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      out.data_ptr<int8_t>(), LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,      // <- reference to matrix C on device
      out_ref,      // <- reference to matrix D on device
      {alpha, beta}, 1};
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement, status: " +
                             std::to_string((int)status));
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize, status: " +
                             std::to_string((int)status));
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot run, status: " +
                             std::to_string((int)status));
  }

#ifdef USE_TORCH_RELU
#undef USE_TORCH_RELU
  out = torch::relu(out);
#endif

  return out;
}

// int8 * int8 = int32
// from https://github.com/IST-DASLab/QUIK/blob/master/src/matmul/matmul.cu
torch::Tensor int8Matmul(torch::Tensor &A, torch::Tensor &B) {
  torch::checkAllSameGPU("int8Matmul", {{A, "A", 0}, {B, "B", 1}});
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);  // 4bit packing is on the columns
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt32).device(A.device()));

  using Gemm = cutlass::gemm::device::Gemm<
      int8_t,                          // ElementA
      cutlass::layout::RowMajor,       // LayoutA
      int8_t,                          // ElementB
      cutlass::layout::ColumnMajor,    // LayoutB
      int32_t,                         // ElementOutput
      cutlass::layout::RowMajor,       // LayoutOutput
      int32_t,                         // ElementAccumulator
      cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
      cutlass::arch::Sm80  // tag indicating target GPU compute architecture
      >;

  Gemm gemmOp;

  using GemmCoord = cutlass::gemm::GemmCoord;

  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {A.data_ptr<int8_t>(), K},
      {B.data_ptr<int8_t>(), K},
      {C.data_ptr<int32_t>(), N},
      {C.data_ptr<int32_t>(), N},
      {1, 0}};

  auto status = gemmOp(arguments);

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return C;
}

// int32 * int32 = int64
// NOTE: Tensor Core is not supported here.
// from https://github.com/IST-DASLab/QUIK/blob/master/src/matmul/matmul.cu
torch::Tensor int32Matmul(torch::Tensor &A, torch::Tensor &B) {
  torch::checkAllSameGPU("int32Matmul", {{A, "A", 0}, {B, "B", 1}});
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);  // 4bit packing is on the columns
  auto C = torch::empty({M, N}, torch::dtype(torch::kInt64).device(A.device()));

  using Gemm = cutlass::gemm::device::Gemm<
      int32_t,                          // ElementA
      cutlass::layout::RowMajor,       // LayoutA
      int32_t,                          // ElementB
      cutlass::layout::ColumnMajor,    // LayoutB
      int64_t,                         // ElementOutput
      cutlass::layout::RowMajor,       // LayoutOutput
      int64_t,                         // ElementAccumulator
      cutlass::arch::OpClassSimt,  // tag indicating Tensor Cores
      cutlass::arch::Sm80 // tag indicating target GPU compute architecture
      >;

  Gemm gemmOp;

  using GemmCoord = cutlass::gemm::GemmCoord;

  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {A.data_ptr<int32_t>(), K},
      {B.data_ptr<int32_t>(), K},
      {C.data_ptr<int64_t>(), N},
      {C.data_ptr<int64_t>(), N},
      {1, 0}};

  auto status = gemmOp(arguments);

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return C;
}