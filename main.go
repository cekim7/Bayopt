package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"math"
	"net/http"
	"sync"

	"gonum.org/v1/gonum/mat"
)

type GP[T any] struct {
	X      []T
	Y      []float64
	Length float64
	Sigma  float64
	Noise  float64
	Kernel func(x1, x2 T) float64

	kinv *mat.SymDense
}

func NewGP[T any](length, sigma, noise float64, kernel func(x1, x2 T) float64) *GP[T] {
	return &GP[T]{Length: length, Sigma: sigma, Noise: noise, Kernel: kernel}
}

func ByteKernel(length, sigma float64) func(x1, x2 []byte) float64 {
	return func(x1, x2 []byte) float64 {
		// A very simplistic structural similarity: differences in length + naive elementwise differences
		var dSq float64
		l1, l2 := len(x1), len(x2)
		diff := float64(l1 - l2)
		dSq += diff * diff * 0.001

		minLen := l1
		if l2 < minLen {
			minLen = l2
		}

		// Subsample to avoid huge kernel computation time on large files
		step := minLen / 100
		if step == 0 {
			step = 1
		}

		var valDiff float64
		for i := 0; i < minLen; i += step {
			d := float64(x1[i]) - float64(x2[i])
			valDiff += d * d
		}
		dSq += valDiff * 0.001

		return sigma * sigma * math.Exp(-0.5*dSq/(length*length))
	}
}

func imageObjective(data []byte) float64 {
	// Dummy objective: just based on size modulo 100
	return float64(len(data) % 100)
}

func audioObjective(data []byte) float64 {
	// Dummy objective: based on sum of bytes
	var sum float64
	for _, b := range data {
		sum += float64(b)
	}
	return math.Mod(sum, 100.0)
}

func RBFKernel(length, sigma float64) func(x1, x2 []float64) float64 {
	return func(x1, x2 []float64) float64 {
		var dSq float64
		for i := 0; i < len(x1) && i < len(x2); i++ {
			d := x1[i] - x2[i]
			dSq += d * d
		}
		return sigma * sigma * math.Exp(-0.5*dSq/(length*length))
	}
}

func (gp *GP[T]) Fit(X []T, Y []float64) {
	gp.X = X
	gp.Y = Y
	n := len(X)
	if n == 0 {
		return
	}
	K := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			val := gp.Kernel(X[i], X[j])
			if i == j {
				val += gp.Noise
			}
			K.SetSym(i, j, val)
		}
	}
	var chol mat.Cholesky
	ok := chol.Factorize(K)
	if !ok {
		for i := 0; i < n; i++ {
			K.SetSym(i, i, K.At(i, i)+1e-4)
		}
		chol.Factorize(K)
	}
	var inv mat.SymDense
	if err := chol.InverseTo(&inv); err != nil {
		log.Println("Inverse error:", err)
	}
	gp.kinv = &inv
}

func (gp *GP[T]) Predict(Xstar []T) ([]float64, []float64) {
	n := len(gp.X)
	m := len(Xstar)
	mean := make([]float64, m)
	std := make([]float64, m)

	if n == 0 {
		for i := 0; i < m; i++ {
			mean[i] = 0
			std[i] = math.Sqrt(gp.Kernel(Xstar[i], Xstar[i]))
		}
		return mean, std
	}

	Ymat := mat.NewVecDense(n, gp.Y)
	alpha := mat.NewVecDense(n, nil)
	alpha.MulVec(gp.kinv, Ymat)

	for i := 0; i < m; i++ {
		kstar := make([]float64, n)
		for j := 0; j < n; j++ {
			kstar[j] = gp.Kernel(Xstar[i], gp.X[j])
		}
		kstarVec := mat.NewVecDense(n, kstar)

		mean[i] = mat.Dot(kstarVec, alpha)

		tmp := mat.NewVecDense(n, nil)
		tmp.MulVec(gp.kinv, kstarVec)
		varRed := mat.Dot(kstarVec, tmp)
		v := gp.Kernel(Xstar[i], Xstar[i]) - varRed
		if v < 0 {
			v = 0
		}
		std[i] = math.Sqrt(v)
	}
	return mean, std
}

// Objective function
func objective(x []float64) float64 {
	// Simulate semiconductor process yield based on chamber temperature
	t := x[0]
	// Optimal temperature around 75C
	baseYield := 95.0
	penalty := 0.01 * (t - 75.0) * (t - 75.0)
	noise := math.Sin((t-75.0)/3.0) * 2.0
	return baseYield - penalty + noise
}

type MediaState struct {
	sync.Mutex
	ObsX [][]byte    `json:"-"`
	ObsY []float64   `json:"obsY"`
	GP   *GP[[]byte] `json:"-"`
}

func (s *MediaState) Reset() {
	s.Lock()
	defer s.Unlock()
	s.ObsX = [][]byte{}
	s.ObsY = []float64{}
	s.GP = NewGP(10.0, 20.0, 1e-1, ByteKernel(10.0, 20.0))
}

func (s *MediaState) Add(data []byte, val float64) {
	s.Lock()
	defer s.Unlock()
	s.ObsX = append(s.ObsX, data)
	s.ObsY = append(s.ObsY, val)
	s.GP.Fit(s.ObsX, s.ObsY)
}

var imageState *MediaState
var audioState *MediaState

type AppState struct {
	sync.Mutex
	ObsX [][]float64    `json:"obsX"`
	ObsY []float64      `json:"obsY"`
	GP   *GP[[]float64] `json:"-"`

	// For visualization
	GridX [][]float64 `json:"gridX"`
	Mean  []float64   `json:"mean"`
	Std   []float64   `json:"std"`
	Acq   []float64   `json:"acq"`
}

var globalState *AppState

func (s *AppState) Reset() {
	s.Lock()
	defer s.Unlock()
	s.ObsX = [][]float64{{20.0}}
	s.ObsY = []float64{objective(s.ObsX[0])}
	s.GP = NewGP(10.0, 20.0, 1e-1, RBFKernel(10.0, 20.0))
	s.GP.Fit(s.ObsX, s.ObsY)
	s.updateGrid()
}

func (s *AppState) updateGrid() {
	grid := make([][]float64, 200)
	for i := range grid {
		// Temperature bounds: 20 to 120 °C
		grid[i] = []float64{20.0 + float64(i)*100.0/199.0}
	}
	s.GridX = grid
	mean, std := s.GP.Predict(grid)
	s.Mean = mean
	s.Std = std

	kappa := 2.0
	acq := make([]float64, len(grid))
	for i := range grid {
		acq[i] = mean[i] + kappa*std[i]
	}
	s.Acq = acq
}

func (s *AppState) Step() {
	s.Lock()
	defer s.Unlock()

	if len(s.GridX) == 0 {
		s.updateGrid()
	}

	bestX := s.GridX[0]
	bestAcq := s.Acq[0]

	for i := 1; i < len(s.GridX); i++ {
		if s.Acq[i] > bestAcq {
			bestAcq = s.Acq[i]
			bestX = s.GridX[i]
		}
	}

	s.ObsX = append(s.ObsX, bestX)
	s.ObsY = append(s.ObsY, objective(bestX))
	s.GP.Fit(s.ObsX, s.ObsY)
	s.updateGrid()
}

func handleUploadImage(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	r.ParseMultipartForm(10 << 20)
	file, _, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "Failed to parse file", http.StatusBadRequest)
		return
	}
	defer file.Close()

	buf := new(bytes.Buffer)
	io.Copy(buf, file)
	data := buf.Bytes()

	val := imageObjective(data)
	imageState.Add(data, val)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(imageState)
}

func handleUploadAudio(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	r.ParseMultipartForm(10 << 20)
	file, _, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "Failed to parse file", http.StatusBadRequest)
		return
	}
	defer file.Close()

	buf := new(bytes.Buffer)
	io.Copy(buf, file)
	data := buf.Bytes()

	val := audioObjective(data)
	audioState.Add(data, val)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(audioState)
}

func handleState(w http.ResponseWriter, r *http.Request) {
	globalState.Lock()
	defer globalState.Unlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(globalState)
}

func handleStep(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	globalState.Step()
	handleState(w, r)
}

func handleReset(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	globalState.Reset()

	imageState = &MediaState{}
	imageState.Reset()
	audioState = &MediaState{}
	audioState.Reset()

	handleState(w, r)
}

func main() {
	globalState = &AppState{}
	globalState.Reset()

	imageState = &MediaState{}
	imageState.Reset()
	audioState = &MediaState{}
	audioState.Reset()

	http.Handle("/", http.FileServer(http.Dir(".")))
	http.HandleFunc("/api/state", handleState)
	http.HandleFunc("/api/step", handleStep)
	http.HandleFunc("/api/reset", handleReset)

	http.HandleFunc("/api/upload/image", handleUploadImage)
	http.HandleFunc("/api/upload/audio", handleUploadAudio)

	log.Println("Listening on :8080...")
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		log.Fatal(err)
	}
}
