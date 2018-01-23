package main

import (
    "fmt"
    "os"
    "github.com/lukechampine/jsteg"
)

func main() {
    // read hidden data:
    f, _ := os.Open("stego.jpg")
    data := []byte("my secret data")
    hidden, _ := jsteg.Reveal(f)
    fmt.Printf(string(hidden[:len(data)]))
}