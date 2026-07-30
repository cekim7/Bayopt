package main

import (
	"encoding/json"
	"log"
	"math"
	"net/http"
	"sync"

	"gonum.org/v1/gonum/mat"
)

type GP struct {
	X      [][]float64
	Y      []float64
	Length float64
	Sigma  float64
	Noise  float64

	kinv *mat.SymDense
}

func NewGP(length, sigma, noise float64) *GP {
	return &GP{Length: length, Sigma: sigma, Noise: noise}
}

func (gp *GP) kernel(x1, x2 []float64) float64 {
	var dSq float64
	for i := 0; i < len(x1) && i < len(x2); i++ {
		d := x1[i] - x2[i]
		dSq += d * d
	}
	return gp.Sigma * gp.Sigma * math.Exp(-0.5*dSq/(gp.Length*gp.Length))
}

func (gp *GP) Fit(X [][]float64, Y []float64) {
	gp.X = X
	gp.Y = Y
	n := len(X)
	if n == 0 {
		return
	}
	K := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			val := gp.kernel(X[i], X[j])
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

func (gp *GP) Predict(Xstar [][]float64) ([]float64, []float64) {
	n := len(gp.X)
	m := len(Xstar)
	mean := make([]float64, m)
	std := make([]float64, m)

	if n == 0 {
		for i := 0; i < m; i++ {
			mean[i] = 0
			std[i] = math.Sqrt(gp.kernel(Xstar[i], Xstar[i]))
		}
		return mean, std
	}

	Ymat := mat.NewVecDense(n, gp.Y)
	alpha := mat.NewVecDense(n, nil)
	alpha.MulVec(gp.kinv, Ymat)

	for i := 0; i < m; i++ {
		kstar := make([]float64, n)
		for j := 0; j < n; j++ {
			kstar[j] = gp.kernel(Xstar[i], gp.X[j])
		}
		kstarVec := mat.NewVecDense(n, kstar)

		mean[i] = mat.Dot(kstarVec, alpha)

		tmp := mat.NewVecDense(n, nil)
		tmp.MulVec(gp.kinv, kstarVec)
		varRed := mat.Dot(kstarVec, tmp)
		v := gp.kernel(Xstar[i], Xstar[i]) - varRed
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

type AppState struct {
	sync.Mutex
	ObsX [][]float64 `json:"obsX"`
	ObsY []float64   `json:"obsY"`
	GP   *GP         `json:"-"`

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
	s.GP = NewGP(10.0, 20.0, 1e-1)
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
	handleState(w, r)
}

func main() {
	globalState = &AppState{}
	globalState.Reset()

	http.Handle("/", http.FileServer(http.Dir(".")))
	http.HandleFunc("/api/state", handleState)
	http.HandleFunc("/api/step", handleStep)
	http.HandleFunc("/api/reset", handleReset)

	log.Println("Listening on :8080...")
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		log.Fatal(err)
	}
}
