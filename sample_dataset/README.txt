Sample Dataset for Dual-Channel Driver Fatigue Detection

Data Sources:
- DDD (visual)
- DD-Database (physio)
- UL-DD (fusion calibration reference)

Structure:
- visual/visual_features.csv
- physio/physio_features.csv
- fusion/fused_table.csv
- metadata/*.txt (columns + quality summary)

Label Definitions:
- visual label: 0=non-drowsy, 1=drowsy
- physio label: 0=alert, 1=drowsy
- fusion y_binary: 0=alert, 1=drowsy

Preprocessing Summary:
- visual: face landmark extraction -> EAR/pitch/yaw -> remove invalid samples
- physio: ECG windowing (60s, step 30s) -> HRV extraction -> remove failed windows
- fusion: align visual/physio outputs by time window and keep valid paired probabilities
