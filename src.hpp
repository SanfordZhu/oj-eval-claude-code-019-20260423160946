#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Step 1: Build K and V matrices by concatenating the first i+1 keys/values
    Matrix* K = matrix_memory_allocator.Allocate("K_" + std::to_string(i));
    Matrix* V = matrix_memory_allocator.Allocate("V_" + std::to_string(i));

    // Start with first key and value
    gpu_sim.Copy(keys[0], K, Position::kInGpuHbm);
    gpu_sim.Copy(values[0], V, Position::kInGpuHbm);

    // Concatenate remaining keys and values
    for (size_t j = 1; j <= i; ++j) {
      Matrix* temp_K = matrix_memory_allocator.Allocate("temp_K_" + std::to_string(j));
      Matrix* temp_V = matrix_memory_allocator.Allocate("temp_V_" + std::to_string(j));

      gpu_sim.Concat(K, keys[j], temp_K, 0, Position::kInGpuHbm);
      gpu_sim.Concat(V, values[j], temp_V, 0, Position::kInGpuHbm);

      gpu_sim.ReleaseMatrix(K);
      gpu_sim.ReleaseMatrix(V);

      K = temp_K;
      V = temp_V;
    }

    // Step 2: Move matrices to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(K);
    gpu_sim.MoveMatrixToSharedMem(V);

    // Step 3: Compute attention
    // Transpose K for matrix multiplication
    Matrix* K_T = matrix_memory_allocator.Allocate("K_T_" + std::to_string(i));
    gpu_sim.Copy(K, K_T, Position::kInSharedMemory);
    gpu_sim.Transpose(K_T, Position::kInSharedMemory);

    // Compute Q * K^T
    Matrix* scores = matrix_memory_allocator.Allocate("scores_" + std::to_string(i));
    gpu_sim.MatMul(current_query, K_T, scores);

    // Apply softmax row-wise
    size_t rows = current_query->GetRowNum();

    // Compute exp of scores
    Matrix* exp_scores = matrix_memory_allocator.Allocate("exp_scores_" + std::to_string(i));
    gpu_sim.MatExp(scores, exp_scores);

    // Normalize each row
    Matrix* attention_weights = matrix_memory_allocator.Allocate("attention_weights_" + std::to_string(i));

    for (size_t row = 0; row < rows; ++row) {
      // Get row from exp_scores
      Matrix* row_vec = matrix_memory_allocator.Allocate("row_vec_" + std::to_string(row));
      gpu_sim.GetRow(exp_scores, row, row_vec, Position::kInSharedMemory);

      // Compute sum of row
      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum_" + std::to_string(row));
      gpu_sim.Sum(row_vec, row_sum);

      // Normalize row
      Matrix* normalized_row = matrix_memory_allocator.Allocate("normalized_row_" + std::to_string(row));
      gpu_sim.MatDiv(row_vec, row_sum, normalized_row);

      // Build attention weights matrix
      if (row == 0) {
        gpu_sim.Copy(normalized_row, attention_weights, Position::kInSharedMemory);
      } else {
        Matrix* temp_weights = matrix_memory_allocator.Allocate("temp_weights_" + std::to_string(row));
        gpu_sim.Concat(attention_weights, normalized_row, temp_weights, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(attention_weights);
        attention_weights = temp_weights;
      }

      // Clean up
      gpu_sim.ReleaseMatrix(row_vec);
      gpu_sim.ReleaseMatrix(row_sum);
      gpu_sim.ReleaseMatrix(normalized_row);
    }

    // Compute attention output: attention_weights * V
    Matrix* output = matrix_memory_allocator.Allocate("output_" + std::to_string(i));
    gpu_sim.MatMul(attention_weights, V, output);

    // Move to HBM and commit
    gpu_sim.MoveMatrixToGpuHbm(output);

    // Clean up
    gpu_sim.ReleaseMatrix(K);
    gpu_sim.ReleaseMatrix(V);
    gpu_sim.ReleaseMatrix(K_T);
    gpu_sim.ReleaseMatrix(scores);
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