"""
對 backtest2_main.py 做兩處修正：
  1. K_KNN = 9 → 7（與 Phase 2 D+3 版最佳 k 一致）
  2. make_svm() 加 class_weight='balanced'（與 Phase 2 SVM v3 一致）

用法：
  python3 patch_backtest.py
"""

import re
from pathlib import Path

target = Path(__file__).parent / "backtest2_main.py"
text   = target.read_text(encoding="utf-8")

# ── 修正 1：K_KNN ──────────────────────────────────────────
text_new = re.sub(r'K_KNN\s*=\s*9', 'K_KNN          = 7', text)
if 'K_KNN          = 7' in text_new:
    print("✓ K_KNN 已從 9 改為 7")
else:
    print("⚠ K_KNN 修改失敗，請手動將 'K_KNN = 9' 改為 'K_KNN = 7'")

# ── 修正 2：make_svm() 加 class_weight='balanced' ──────────
old_svm = "return LinearSVC(C=5.0, max_iter=5000, random_state=42)"
new_svm = "return LinearSVC(C=5.0, max_iter=5000, random_state=42, class_weight='balanced')"

if old_svm in text_new:
    text_new = text_new.replace(old_svm, new_svm)
    print("✓ make_svm() 已加入 class_weight='balanced'")
elif new_svm in text_new:
    print("✓ make_svm() 已有 class_weight='balanced'，無需修改")
else:
    print("⚠ make_svm() 修改失敗，請手動在 LinearSVC 裡加上 class_weight='balanced'")

target.write_text(text_new, encoding="utf-8")
print(f"\n完成，已寫回：{target}")
