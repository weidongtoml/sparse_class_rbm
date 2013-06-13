package rbm

import (
	"math"
)

// Calculate P(y = 1|X)
// Note instance.y is ignored in the calcuation.
func (rbm *SparseClassRBM) GetPrediction(instance *DataInstance) float32 {
	energy_y_1 := float64(rbm.d)
	energy_y_0 := float64(0.0)
	
	for h, c_val := range rbm.c {
		sum_of_x_weights := float64(c_val)
		for x_class, x_value := range instance.x {
			sum_of_x_weights += float64(rbm.w[x_class][h][x_value])
		}
		energy_y_1 += float64(softplus(sum_of_x_weights + float64(rbm.u[h])))
		energy_y_0 += float64(softplus(sum_of_x_weights))
	}
	
	return float32(math.Exp(energy_y_1) / (math.Exp(energy_y_1) + math.Exp(energy_y_0)))
}

func softplus(x float64) float64 {
	return math.Log(1 + x)
}
