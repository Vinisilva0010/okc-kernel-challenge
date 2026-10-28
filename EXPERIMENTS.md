# Experiment log

Every judge run, with the hypothesis and the measured result. Times in µs (verify full: batch 16 / batch 32).

| version | job | hypothesis | result | verdict |
|---|---|---|---|---|
| sub_v5 | 6821872e91094b4c | infraax sonnet_v5 as starting point | 1.788x; verify full 164.9 / 325.8 | baseline |
| sub_v6 | 895aca84535640fd | P.V as FP8 MMA (P x 2^8, as FA3) removes the V FP8->FP16 convert | 1.555x; verify full 245.6 / 483.9; 16/16 tests passed | rejected |

## Lessons

- v6: in Triton, an FP8 P.V dot is slower than FP16. The FP32 accumulator layout of an FP8 MMA does not match the FP8 A-operand layout, so Triton routes P through shared memory every tile; FA3 avoids this with in-register byte permutes. FP8 P.V needs hand-written CUDA.
