package vectorize

import "time"

const (
	maxAmount           = 10000.0
	maxInstallments     = 12.0
	amountVsAvgRatio    = 10.0
	maxMinutes          = 1440.0
	maxKm               = 1000.0
	maxTxCount24h       = 20.0
	maxMerchantAvgAmount = 10000.0
)

// mccRisk maps MCC codes to risk scores. Default is 0.5.
var mccRisk = map[string]float32{
	"5411": 0.15,
	"5812": 0.30,
	"5912": 0.20,
	"5944": 0.45,
	"7801": 0.80,
	"7802": 0.75,
	"7995": 0.85,
	"4511": 0.35,
	"5311": 0.25,
	"5999": 0.50,
}

// Request mirrors the POST /fraud-score payload.
type Request struct {
	ID          string      `json:"id"`
	Transaction Transaction `json:"transaction"`
	Customer    Customer    `json:"customer"`
	Merchant    Merchant    `json:"merchant"`
	Terminal    Terminal    `json:"terminal"`
	LastTx      *LastTx     `json:"last_transaction"`
}

type Transaction struct {
	Amount      float32 `json:"amount"`
	Installments int    `json:"installments"`
	RequestedAt string  `json:"requested_at"`
}

type Customer struct {
	AvgAmount       float32  `json:"avg_amount"`
	TxCount24h      int      `json:"tx_count_24h"`
	KnownMerchants  []string `json:"known_merchants"`
}

type Merchant struct {
	ID        string  `json:"id"`
	MCC       string  `json:"mcc"`
	AvgAmount float32 `json:"avg_amount"`
}

type Terminal struct {
	IsOnline    bool    `json:"is_online"`
	CardPresent bool    `json:"card_present"`
	KmFromHome  float32 `json:"km_from_home"`
}

type LastTx struct {
	Timestamp     string  `json:"timestamp"`
	KmFromCurrent float32 `json:"km_from_current"`
}

func clamp(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// Vectorize converts a Request into a 14-dimensional float32 vector.
func Vectorize(r *Request) [14]float32 {
	var v [14]float32

	v[0] = clamp(r.Transaction.Amount / maxAmount)
	v[1] = clamp(float32(r.Transaction.Installments) / maxInstallments)
	v[2] = clamp((r.Transaction.Amount / r.Customer.AvgAmount) / amountVsAvgRatio)

	t, _ := time.Parse(time.RFC3339, r.Transaction.RequestedAt)
	v[3] = float32(t.UTC().Hour()) / 23.0
	v[4] = float32((int(t.UTC().Weekday()) + 6) % 7) / 6.0 // Mon=0 Sun=6

	if r.LastTx != nil {
		lastT, _ := time.Parse(time.RFC3339, r.LastTx.Timestamp)
		minutes := float32(t.Sub(lastT).Minutes())
		v[5] = clamp(minutes / maxMinutes)
		v[6] = clamp(r.LastTx.KmFromCurrent / maxKm)
	} else {
		v[5] = -1
		v[6] = -1
	}

	v[7] = clamp(r.Terminal.KmFromHome / maxKm)
	v[8] = clamp(float32(r.Customer.TxCount24h) / maxTxCount24h)

	if r.Terminal.IsOnline {
		v[9] = 1
	}
	if r.Terminal.CardPresent {
		v[10] = 1
	}

	v[11] = 1
	for _, m := range r.Customer.KnownMerchants {
		if m == r.Merchant.ID {
			v[11] = 0
			break
		}
	}

	if risk, ok := mccRisk[r.Merchant.MCC]; ok {
		v[12] = risk
	} else {
		v[12] = 0.5
	}

	v[13] = clamp(r.Merchant.AvgAmount / maxMerchantAvgAmount)

	return v
}
