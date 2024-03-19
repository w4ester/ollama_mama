package quants

import (
	"context"
	"fmt"
	"math/rand"
	"os/exec"
	"path/filepath"
)

// usage: quantize [--help] [--allow-requantize] [--leave-output-tensor] [--pure] [--imatrix] [--include-weights] [--exclude-weights] model-f32.gguf [model-quant.gguf] type [nthreads]

//   --allow-requantize: Allows requantizing tensors that have already been quantized. Warning: This can severely reduce quality compared to quantizing from 16bit or 32bit
//   --leave-output-tensor: Will leave output.weight un(re)quantized. Increases model size but may also increase quality, especially when requantizing
//   --pure: Disable k-quant mixtures and quantize all tensors to the same type
//   --imatrix file_name: use data in file_name as importance matrix for quant optimizations
//   --include-weights tensor_name: use importance matrix for this/these tensor(s)
//   --exclude-weights tensor_name: use importance matrix for this/these tensor(s)
// Note: --include-weights and --exclude-weights cannot be used together

// Allowed quantization types:
//    2  or  Q4_0    :  3.56G, +0.2166 ppl @ LLaMA-v1-7B
//    3  or  Q4_1    :  3.90G, +0.1585 ppl @ LLaMA-v1-7B
//    8  or  Q5_0    :  4.33G, +0.0683 ppl @ LLaMA-v1-7B
//    9  or  Q5_1    :  4.70G, +0.0349 ppl @ LLaMA-v1-7B
//   19  or  IQ2_XXS :  2.06 bpw quantization
//   20  or  IQ2_XS  :  2.31 bpw quantization
//   28  or  IQ2_S   :  2.5  bpw quantization
//   29  or  IQ2_M   :  2.7  bpw quantization
//   24  or  IQ1_S   :  1.56 bpw quantization
//   10  or  Q2_K    :  2.63G, +0.6717 ppl @ LLaMA-v1-7B
//   21  or  Q2_K_S  :  2.16G, +9.0634 ppl @ LLaMA-v1-7B
//   23  or  IQ3_XXS :  3.06 bpw quantization
//   26  or  IQ3_S   :  3.44 bpw quantization
//   27  or  IQ3_M   :  3.66 bpw quantization mix
//   12  or  Q3_K    : alias for Q3_K_M
//   22  or  IQ3_XS  :  3.3 bpw quantization
//   11  or  Q3_K_S  :  2.75G, +0.5551 ppl @ LLaMA-v1-7B
//   12  or  Q3_K_M  :  3.07G, +0.2496 ppl @ LLaMA-v1-7B
//   13  or  Q3_K_L  :  3.35G, +0.1764 ppl @ LLaMA-v1-7B
//   25  or  IQ4_NL  :  4.50 bpw non-linear quantization
//   30  or  IQ4_XS  :  4.25 bpw non-linear quantization
//   15  or  Q4_K    : alias for Q4_K_M
//   14  or  Q4_K_S  :  3.59G, +0.0992 ppl @ LLaMA-v1-7B
//   15  or  Q4_K_M  :  3.80G, +0.0532 ppl @ LLaMA-v1-7B
//   17  or  Q5_K    : alias for Q5_K_M
//   16  or  Q5_K_S  :  4.33G, +0.0400 ppl @ LLaMA-v1-7B
//   17  or  Q5_K_M  :  4.45G, +0.0122 ppl @ LLaMA-v1-7B
//   18  or  Q6_K    :  5.15G, +0.0008 ppl @ LLaMA-v1-7B
//    7  or  Q8_0    :  6.70G, +0.0004 ppl @ LLaMA-v1-7B
//    1  or  F16     : 13.00G              @ 7B
//    0  or  F32     : 26.00G              @ 7B
//           COPY    : only copy tensors, no quantizing

// Convert runs the quantize command for the file in and given level, and
// then returns the output file, or an error if any.
func Convert(ctx context.Context, workDir, in, level string) (out string, err error) {
	out = filepath.Join(workDir, generateRandomString())
	cmd := exec.Command("quantize",
		in,
		out,
		level,
	)
	b, err := cmd.CombinedOutput()
	if err != nil {
		// TODO(bmizerany): temp hack; better error handling / messages
		return "", fmt.Errorf("quantize: %v: %s", err, b)
	}
	return out, nil
}

func generateRandomString() string {
	var letterRunes = []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
	b := make([]rune, 10)
	for i := range b {
		b[i] = letterRunes[rand.Intn(len(letterRunes))]
	}
	return string(b)
}
