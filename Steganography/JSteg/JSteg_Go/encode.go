package main

import (
    "fmt"
    "image/jpeg"
    "os"
    "github.com/lukechampine/jsteg"
)

func main() {
    f, _ := os.Open("cover.jpg")
    img, _ := jpeg.Decode(f)
    out, _ := os.Create("stego.jpg")
    data := []byte("my secret data")
    jsteg.Hide(out, img, data, nil)
}