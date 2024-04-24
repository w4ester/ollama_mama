package main

import (
	"fmt"

	"github.com/ollama/ollama/llama"
)

func main() {
	fmt.Println(llama.SystemInfo())
	llama.Run("gemma.bin", "Hello, world!")
}
