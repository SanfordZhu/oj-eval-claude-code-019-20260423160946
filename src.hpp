#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Step 1: Concatenate K and V matrices for the first i+1 elements
    Matrix* concat_k = matrix_memory_allocator.Allocate("concat_k_" + std::to_string(i));
    Matrix* concat_v = matrix_memory_allocator.Allocate("concat_v_" + std::to_string(i));

    // Start with the first key and value
    gpu_sim.Copy(keys[0], concat_k, Position::kInGpuHbm);
    gpu_sim.Copy(values[0], concat_v, Position::kInGpuHbm);

    // Concatenate remaining keys and values
    for (size_t j = 1; j <= i; ++j) {
      Matrix* temp_k = matrix_memory_allocator.Allocate("temp_k_" + std::to_string(j));
      Matrix* temp_v = matrix_memory_allocator.Allocate("temp_v_" + std::to_string(j));

      gpu_sim.Concat(concat_k, keys[j], temp_k, 0, Position::kInGpuHbm);
      gpu_sim.Concat(concat_v, values[j], temp_v, 0, Position::kInGpuHbm);

      gpu_sim.ReleaseMatrix(concat_k);
      gpu_sim.ReleaseMatrix(concat_v);

      concat_k = temp_k;
      concat_v = temp_v;
    }

    // Step 2: Move matrices to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(concat_k);
    gpu_sim.MoveMatrixToSharedMem(concat_v);

    // Step 3: Compute attention
    // Transpose K for matrix multiplication
    Matrix* k_transpose = matrix_memory_allocator.Allocate("k_transpose_" + std::to_string(i));
    gpu_sim.Copy(concat_k, k_transpose, Position::kInSharedMemory);
    gpu_sim.Transpose(k_transpose, Position::kInSharedMemory);

    // Compute Q * K^T
    Matrix* attention_scores = matrix_memory_allocator.Allocate("attention_scores_" + std::to_string(i));
    gpu_sim.MatMul(current_query, k_transpose, attention_scores);

    // Apply softmax row-wise
    size_t rows = current_query->GetRowNum();

    // Compute exp for all elements
    Matrix* exp_scores = matrix_memory_allocator.Allocate("exp_scores_" + std::to_string(i));
    gpu_sim.MatExp(attention_scores, exp_scores);

    // Compute row sums and normalize
    Matrix* attention_weights = matrix_memory_allocator.Allocate("attention_weights_" + std::to_string(i));

    for (size_t row = 0; row < rows; ++row) {
      // Get the row
      Matrix* row_vec = matrix_memory_allocator.Allocate("row_vec_" + std::to_string(row));
      gpu_sim.GetRow(exp_scores, row, row_vec, Position::kInSharedMemory);

      // Compute sum of the row
      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum_" + std::to_string(row));
      gpu_sim.Sum(row_vec, row_sum);

      // Normalize the row
      Matrix* normalized_row = matrix_memory_allocator.Allocate("normalized_row_" + std::to_string(row));
      gpu_sim.MatDiv(row_vec, row_sum, normalized_row);

      // Build the attention weights matrix row by row
      if (row == 0) {
        gpu_sim.Copy(normalized_row, attention_weights, Position::kInSharedMemory);
      } else {
        Matrix* temp_weights = matrix_memory_allocator.Allocate("temp_weights_" + std::to_string(row));
        gpu_sim.Concat(attention_weights, normalized_row, temp_weights, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(attention_weights);
        attention_weights = temp_weights;
      }

      // Clean up row-specific matrices
      gpu_sim.ReleaseMatrix(row_vec);
      gpu_sim.ReleaseMatrix(row_sum);
      gpu_sim.ReleaseMatrix(normalized_row);
    }

    // Step 4: Multiply attention weights with V
    Matrix* output = matrix_memory_allocator.Allocate("output_" + std::to_string(i));
    gpu_sim.MatMul(attention_weights, concat_v, output);

    // Step 5: Move result to HBM and commit
    gpu_sim.MoveMatrixToGpuHbm(output);

    // Clean up intermediate matrices
    gpu_sim.ReleaseMatrix(concat_k);
    gpu_sim.ReleaseMatrix(concat_v);
    gpu_sim.ReleaseMatrix(k_transpose);
    gpu_sim.ReleaseMatrix(attention_scores);
    gpu_sim.ReleaseMatrix(exp_scores);
    gpu_sim.ReleaseMatrix(attention_weights);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*output);
  
    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu