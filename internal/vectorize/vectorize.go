package vectorize

import "time"

const (
	maxAmount            = 10000.0
	maxInstallments      = 12.0
	amountVsAvgRatio     = 10.0
	maxMinutes           = 1440.0
	maxKm                = 1000.0
	maxTxCount24h        = 20.0
	maxMerchantAvgAmount = 10000.0
)

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
	Amount       float32 `json:"amount"`
	Installments int     `json:"installments"`
	RequestedAt  string  `json:"requested_at"`
}

type Customer struct {
	AvgAmount      float32  `json:"avg_amount"`
	TxCount24h     int      `json:"tx_count_24h"`
	KnownMerchants []string `json:"known_merchants"`
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

func two(s string, i int) int {
	return int(s[i]-'0')*10 + int(s[i+1]-'0')
}

func four(s string, i int) int {
	return int(s[i]-'0')*1000 + int(s[i+1]-'0')*100 + int(s[i+2]-'0')*10 + int(s[i+3]-'0')
}

func daysFromCivil(y, m, d int) int64 {
	if m <= 2 {
		y--
	}
	era := y / 400
	yoe := y - era*400
	mp := m
	if mp > 2 {
		mp -= 3
	} else {
		mp += 9
	}
	doy := (153*mp+2)/5 + d - 1
	doe := yoe*365 + yoe/4 - yoe/100 + doy
	return int64(era*146097 + doe - 719468)
}

func parseTimestampFast(s string) (hour int, monday0 int, minuteSerial int64, ok bool) {
	if len(s) < len("2006-01-02T15:04") || s[4] != '-' || s[7] != '-' || s[10] != 'T' || s[13] != ':' {
		return 0, 0, 0, false
	}
	y := four(s, 0)
	m := two(s, 5)
	d := two(s, 8)
	h := two(s, 11)
	min := two(s, 14)
	days := daysFromCivil(y, m, d)
	return h, int((days + 3) % 7), days*1440 + int64(h*60+min), true
}

func mccRiskFast(mcc string) float32 {
	switch mcc {
	case "5411":
		return 0.15
	case "5812":
		return 0.30
	case "5912":
		return 0.20
	case "5944":
		return 0.45
	case "7801":
		return 0.80
	case "7802":
		return 0.75
	case "7995":
		return 0.85
	case "4511":
		return 0.35
	case "5311":
		return 0.25
	case "5999":
		return 0.50
	default:
		return 0.5
	}
}

// Vectorize converts a Request into a 14-dimensional float32 vector.
func Vectorize(r *Request) [14]float32 {
	var v [14]float32

	v[0] = clamp(r.Transaction.Amount / maxAmount)
	v[1] = clamp(float32(r.Transaction.Installments) / maxInstallments)
	v[2] = clamp((r.Transaction.Amount / r.Customer.AvgAmount) / amountVsAvgRatio)

	hour, weekday, txMinute, ok := parseTimestampFast(r.Transaction.RequestedAt)
	if !ok {
		t, _ := time.Parse(time.RFC3339, r.Transaction.RequestedAt)
		hour = t.UTC().Hour()
		weekday = (int(t.UTC().Weekday()) + 6) % 7
		txMinute = t.Unix() / 60
	}
	v[3] = float32(hour) / 23.0
	v[4] = float32(weekday) / 6.0 // Mon=0 Sun=6

	if r.LastTx != nil {
		_, _, lastMinute, ok := parseTimestampFast(r.LastTx.Timestamp)
		if !ok {
			lastT, _ := time.Parse(time.RFC3339, r.LastTx.Timestamp)
			lastMinute = lastT.Unix() / 60
		}
		minutes := float32(txMinute - lastMinute)
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

	v[12] = mccRiskFast(r.Merchant.MCC)

	v[13] = clamp(r.Merchant.AvgAmount / maxMerchantAvgAmount)

	return v
}
