| run | GGUF | size GB | CPU | thr | n | acc GPU fp16 | acc CPU | Δacc | agree | schema ok | prompt tok | out tok | s/case (mean) | p95 s | peak RAM GB | cases/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpu_vm_gemma4_hybrid_q4_k_m_sub100 | gemma4-e4b-rca-Q4_K_M.gguf | 5.30 | Intel(R) Xeon(R) Processor @ 2.80GHz | 4 | 100 | 0.850 | 0.790 | -0.060 | 0.940 | 0.970 | 805 | 219 | 52.6 | 68.4 | 12.17 | 68 |
| cpu_vm_gemma4_hybrid_q8_0_sub100 | gemma4-e4b-rca-Q8_0.gguf | 7.97 | Intel(R) Xeon(R) Processor @ 2.80GHz | 4 | 100 | 0.850 | 0.840 | -0.010 | 0.990 | 1.000 | 805 | 209 | 85.1 | 116.1 | 11.76 | 42 |
| cpu_vm_ministral3_hybrid_q4_k_m_sub100 | ministral3-3b-rca-Q4_K_M.gguf | 2.15 | Intel(R) Xeon(R) Processor @ 2.80GHz | 4 | 100 | 0.770 | 0.760 | -0.010 | 0.930 | 1.000 | 844 | 225 | 44.0 | 63.4 | 8.97 | 82 |
| cpu_kaggle_ministral3_hybrid_q4_k_m/ministral3_hybrid_q4_k_m | ministral3-3b-rca-Q4_K_M.gguf | 2.15 | Intel(R) Xeon(R) CPU @ 2.20GHz | 4 | 308 | 0.656 | 0.649 | -0.006 | 0.935 | 0.997 | 838 | 217 | 64.8 | 102.6 | 12.32 | 56 |
| cpu_kaggle_ministral3_hybrid_q4_k_m/raw_download/ministral3_hybrid_q4_k_m | ministral3-3b-rca-Q4_K_M.gguf | 2.15 | Intel(R) Xeon(R) CPU @ 2.20GHz | 4 | 308 | 0.656 | 0.649 | -0.006 | 0.935 | 0.997 | 838 | 217 | 64.8 | 102.6 | 12.32 | 56 |
| cpu_kaggle_ministral3_hybrid_q8_0/ministral3_hybrid_q8_0 | ministral3-3b-rca-Q8_0.gguf | 3.65 | Intel(R) Xeon(R) CPU @ 2.20GHz | 4 | 308 | 0.656 | 0.672 | +0.016 | 0.984 | 0.994 | 838 | 215 | 76.6 | 115.7 | 12.36 | 47 |
| cpu_kaggle_ministral3_hybrid_q8_0/raw_download/ministral3_hybrid_q8_0 | ministral3-3b-rca-Q8_0.gguf | 3.65 | Intel(R) Xeon(R) CPU @ 2.20GHz | 4 | 308 | 0.656 | 0.672 | +0.016 | 0.984 | 0.994 | 838 | 215 | 76.6 | 115.7 | 12.36 | 47 |
