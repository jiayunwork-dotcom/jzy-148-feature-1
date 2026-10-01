"""生成确定性开发样本（无随机过程，行内规则驱动，可复现）。

5 个特征 + label，约 1200 行，违约率约 25%：
- age       数值，与风险强单调（年龄越大违约率越低），8% 缺失
- income    数值，弱单调，含部分噪声
- debt_ratio 数值，基本无区分度（IV 阈值过滤的验证对象）
- city      类别，城市间风险差异明显
- job_grade 类别，等级越差违约越高，5% 缺失
"""
from __future__ import annotations

import os

N = 1200
HEADER = "label,age,income,debt_ratio,city,job_grade"


def _row(i: int) -> str:
    # 循环生成，保证类别覆盖均衡
    city = ["BJ", "SH", "GZ", "CD", "WH", "XA"][i % 6]
    age = 22 + (i * 7) % 41  # 22..62

    base = 0.0
    # age：越低越容易坏；每 5 岁一档
    base += max(0.0, 0.9 - (age - 22) * 0.022)
    # city：BJ/SH 资质较好，CD/WH 风险高
    base += {"BJ": -0.25, "SH": -0.15, "GZ": 0.0,
             "CD": 0.30, "WH": 0.35, "XA": 0.10}[city]
    # job_grade：A..E 逐级更坏
    grade = "EDCBA"[i % 5]
    base += {"A": -0.20, "B": -0.05, "C": 0.10,
             "D": 0.25, "E": 0.40}[grade]
    # income：由 i 决定的确定性弱信号
    income = 3000 + (i * 337) % 17000
    if income > 12000:
        base -= 0.10
    elif income < 6000:
        base += 0.10
    # debt_ratio：纯噪声，跟 i 奇偶走的对称网格
    debt_ratio = round(0.1 + ((i * 37) % 81) / 100.0, 2)

    # 确定性坏标签：阈值由 i 的模运算微调，避免整齐切面
    threshold = 0.80 + ((i * 13) % 17) / 100.0
    bad = 1 if base >= threshold else 0

    age_s = "" if i % 12 == 0 else str(age)
    grade_s = "" if i % 20 == 0 else grade
    income_s = "" if i % 31 == 0 else str(income)
    return f"{bad},{age_s},{income_s},{debt_ratio},{city},{grade_s}"


def main() -> None:
    out = os.path.join(os.path.dirname(__file__), "dev_sample.csv")
    with open(out, "w", encoding="utf-8") as f:
        f.write(HEADER + "\n")
        for i in range(N):
            f.write(_row(i) + "\n")
    n_bad = sum(1 for i in range(N) if _row(i).startswith("1,"))
    print(f"wrote {out}: {N} rows, {n_bad} bad ({n_bad / N:.3f})")


if __name__ == "__main__":
    main()
