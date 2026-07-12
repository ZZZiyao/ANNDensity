# Veto study matrix -- one line per (structure x size) config; index = SLURM_ARRAY_TASK_ID.
# Format: NAME|VETO_SPEC
# Structure axis: mass (eps flat) / angle-peak (cos~-0.5) / angle-tail (cos~0.7, eps steep) / 2D / 3D
# Size axis: windows widen to remove ~10/20/30/40% of events (mass) or comparable for angles.
VETO_CONFIGS=(
  # --- mass veto (eps flat in mass): 4 sizes ---
  "mass_s|m1:0.82:0.86"
  "mass_m|m1:0.80:0.88"
  "mass_l|m1:0.78:0.90"
  "mass_xl|m1:0.75:0.93"
  # --- angle-peak veto (cos1 near peak -0.5, eps high & gentle): 3 sizes ---
  "apeak_s|cos1:-0.6:-0.4"
  "apeak_m|cos1:-0.7:-0.3"
  "apeak_l|cos1:-0.85:-0.15"
  # --- angle-tail veto (cos1 ~0.7, eps falling steeply -- hardest 1D): 3 sizes ---
  "atail_s|cos1:0.6:0.8"
  "atail_m|cos1:0.5:0.9"
  "atail_l|cos1:0.4:1.0"
  # --- 2D veto (cos1 x cos2: carve a block out of the eps dome, the strongest 2D structure) ---
  "2d_s|cos1:-0.6:-0.2,cos2:-0.6:-0.2"
  "2d_l|cos1:-0.7:0.1,cos2:-0.7:0.1"
  # --- 3D veto (carve a block out of the cos1-cos2 dome + a mass slab): 1 ---
  "3d|cos1:-0.4:0.2,cos2:-0.4:0.2,m1:0.80:0.92"
)
