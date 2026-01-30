# Synergi_final

## CSV requirements
- Columns: `Time;c/kWh`
- Separator: `;`
- Decimal comma is supported (e.g. `12,34`)
- 15-minute timestamps for every slot (00:00, 00:15, ..., 23:45)
- Order can be unsorted; the loader sorts by time

Notes:
- If your data includes DST gaps/overlaps, use UTC timestamps or a range without DST shifts.

## Run
```bash
python3 main.py --input Spot15_month.csv
```

Lock the row count (e.g. 30 days * 96 = 2880):
```bash
python3 main.py --input Spot15_month.csv --expected-rows 2880
```

Audit (if you want detailed checks and per-slot exports):
```bash
python3 main.py --input Spot15_month.csv --audit
```
