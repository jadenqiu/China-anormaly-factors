"""
factor_factory.py
=================
中国股票因子库 - 工厂函数与注册表

职责：
  1. 统一导入所有因子类（主文件 + Qiu 的修订/补充文件）
  2. 构建工厂注册表  REGISTRY: {abbr -> ctor}
  3. 提供工厂函数   create(name, **kwargs) -> BaseFactor
  4. main() 遍历所有因子，依次 calculate -> as_factor_frame -> save
     （含 tqdm 进度条 + 统一输出目录 <root>/clean_output/）
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Type

from tqdm import tqdm

# ---------------------------------------------------------------------------
# 0.  路径设置
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.resolve()
CLEAN_OUTPUT_DIR = ROOT_DIR / "clean_output"
CLEAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1.  导入主因子库
# ---------------------------------------------------------------------------
# 将项目根目录加入 sys.path，确保相对导入可用
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from china_anomalies_factors import (
    BaseFactor,
    # ── A1 Momentum ──
    SueFactor,
    AbrFactor,
    ReFactor,
    R6Factor,
    R11Factor,
    RsFactor,
    TesFactor,
    NeiFactor,
    ImFactor,
    W52Factor,
    Rm6Factor,
    Rm11Factor,
    # ── A2 Value ──
    BmFactor,
    DmFactor,
    AmFactor,
    EpFactor,
    CpFactor,
    RevFactor,
    SpFactor,
    DmqFactor,
    AmqFactor,
    EpqFactor,
    CpqFactor,
    SpqFactor,
    OcpFactor,
    OcpqFactor,
    # ── A3 Investment ──
    AgrFactor,
    IaFactor,
    NsiFactor,
    IaqFactor,
    DpiaFactor,
    NoaFactor,
    DnoaFactor,
    DlnoFactor,
    IgFactor,
    Ig2Factor,
    Ig3Factor,
    # ── A4 Profitability（用 A2 基类）──
    RoeFactor,
    DroeFactor,
    RoaFactor,
    DroaFactor,
    GpaFactor,
    OpaFactor,
    CtoFactor,
    OlaFactor,
    OleFactor,
    GlaFactor,
    CtoqFactor,
    GlaqFactor,
    OleqFactor,
    OlaqFactor,
    OpeFactor,
    CopFactor,
    ClaFactor,
    ClaqFactor,
    # ── A5 Intangibles ──
    AgeFactor,
    DsiFactor,
    AnaFactor,
    RdmFactor,
    RdmqFactor,
    RdsFactor,
    AdmFactor,
    gAdFactor,
    TanFactor,
    # ── A6 Trading Frictions ──
    MeFactor,
    TurnFactor,
    TurFactor,
    DtvFactor,
    PpsFactor,
    AmiFactor,
    SrevFactor,
    TsFactor,
    ShlFactor,
    IvffFactor,
    IvcFactor,
)

# ---------------------------------------------------------------------------
# 2.  导入 Qiu 的修订版因子（Fixed* 重载原始实现）
# ---------------------------------------------------------------------------
from Factor_code_revise_Qiu import (
    FixedReFactor,
    FixedW52Factor,
    FixedRm6Factor,
    FixedRm11Factor,
    FixedCpFactor,
    FixedDlnoFactor,
    FixedRoaFactor,
    FixedCtoFactor,
    FixedOpeFactor,
    FixedOleFactor,
    FixedOleqFactor,
    FixedOlaFactor,
    FixedOlaqFactor,
    FixedCopFactor,
    FixedClaFactor,
    FixedClaqFactor,
    FixedAdmFactor,
    FixedgAdFactor,
    FixedRdmFactor,
    FixedRdsFactor,
    FixedAnaFactor,
)

# ---------------------------------------------------------------------------
# 3.  导入 Qiu 补充/覆盖的因子（transfer 文件）
#
#     transfer 文件包含两类因子：
#       (a) 与主文件同名同 abbr 的重写版：Rs, Tes, Nei, Me, Ivff, Ivc,
#           Tur, Dtv, Pps, Ami, Ts, Srev, Shl  → 以 transfer 版为准，第三轮覆盖
#       (b) 主文件中完全没有的新因子：Bmj, Bmq, Sr, Sg, Cei, Cdi, ...
#
#     统一用 "as T_<Name>" 别名导入所有与主文件同名的类，避免命名空间污染；
#     纯新因子直接导入即可。
# ---------------------------------------------------------------------------
from Factor_code_transfer_Qiu import (
    # ── A1：transfer 重写版（覆盖主文件同 abbr）──
    RsFactor   as T_RsFactor,
    TesFactor  as T_TesFactor,
    NeiFactor  as T_NeiFactor,
    # ── A2：纯新因子 ──
    BmjFactor,
    BmqFactor,
    SrFactor,
    SgFactor,
    # ── A3：纯新因子 ──
    CeiFactor,
    CdiFactor,
    IvgFactor,
    IvcFactor  as T_IvcFactor,   # A3 版 ivc，与主文件 A6 版同 abbr，transfer 优先
    OaFactor,
    TaFactor,
    DWcFactor,
    DCoaFactor,
    DColFactor,
    DNcoFactor,
    DncaFactor,
    DnclFactor,
    DfinFactor,
    DstiFactor,
    DltiFactor,
    DfnlFactor,
    DbeFactor,
    PoaFactor,
    PtaFactor,
    NxfFactor,
    NefFactor,
    NdfFactor,
    # ── A6：transfer 重写版（覆盖主文件同 abbr）──
    MeFactor   as T_MeFactor,
    IvffFactor as T_IvffFactor,
    TurFactor  as T_TurFactor,
    DtvFactor  as T_DtvFactor,
    PpsFactor  as T_PpsFactor,
    AmiFactor  as T_AmiFactor,
    TsFactor   as T_TsFactor,
    SrevFactor as T_SrevFactor,
    ShlFactor  as T_ShlFactor,
    # ── A6：纯新因子 ──
    MdrFactor,
)

# ---------------------------------------------------------------------------
# 4.  工厂注册表
#     key  = 因子 abbr（小写，作为唯一键）
#     value = 类构造器（无参数即可实例化）
#
#     优先级规则（三轮注册，后轮覆盖前轮）：
#       第一轮：主文件原始版
#       第二轮：revise 文件 Fixed* 修订版  → 覆盖第一轮中对应 abbr
#       第三轮：transfer 文件版本（含重写 + 纯新）→ 覆盖前两轮中对应 abbr
#     即最终优先级：transfer > Fixed* revise > 主文件原始
# ---------------------------------------------------------------------------

REGISTRY: Dict[str, Type[BaseFactor]] = {}

def _register(*factor_classes: Type[BaseFactor]) -> None:
    """批量注册因子类到 REGISTRY，以 abbr 小写作为键。"""
    for cls in factor_classes:
        # 实例化一次取 abbr（无副作用）
        try:
            key = cls().abbr.lower()
        except Exception:
            # 如果构造需要参数则跳过，不应出现
            continue
        REGISTRY[key] = cls


# ── 第一轮：主文件原始版 ──
_register(
    # A1
    SueFactor, AbrFactor, ReFactor, R6Factor, R11Factor,
    RsFactor, TesFactor, NeiFactor, ImFactor, W52Factor, Rm6Factor, Rm11Factor,
    # A2
    BmFactor, DmFactor, AmFactor, EpFactor, CpFactor, RevFactor, SpFactor,
    DmqFactor, AmqFactor, EpqFactor, CpqFactor, SpqFactor, OcpFactor, OcpqFactor,
    # A3
    AgrFactor, IaFactor, NsiFactor, IaqFactor, DpiaFactor,
    NoaFactor, DnoaFactor, DlnoFactor, IgFactor, Ig2Factor, Ig3Factor,
    # A4
    RoeFactor, DroeFactor, RoaFactor, DroaFactor, GpaFactor, OpaFactor,
    CtoFactor, OlaFactor, OleFactor, GlaFactor, CtoqFactor, GlaqFactor,
    OleqFactor, OlaqFactor, OpeFactor, CopFactor, ClaFactor, ClaqFactor,
    # A5
    AgeFactor, DsiFactor, AnaFactor, RdmFactor, RdmqFactor, RdsFactor,
    AdmFactor, gAdFactor, TanFactor,
    # A6
    MeFactor, TurnFactor, TurFactor, DtvFactor, PpsFactor, AmiFactor,
    SrevFactor, TsFactor, ShlFactor, IvffFactor, IvcFactor,
)

# ── 第二轮：Qiu 修订版（Fixed*，会覆盖原始版 abbr 对应条目）──
_register(
    FixedReFactor, FixedW52Factor, FixedRm6Factor, FixedRm11Factor,
    FixedCpFactor, FixedDlnoFactor, FixedRoaFactor, FixedCtoFactor,
    FixedOpeFactor, FixedOleFactor, FixedOleqFactor, FixedOlaFactor,
    FixedOlaqFactor, FixedCopFactor, FixedClaFactor, FixedClaqFactor,
    FixedAdmFactor, FixedgAdFactor, FixedRdmFactor, FixedRdsFactor,
    FixedAnaFactor,
)

# ── 第三轮：transfer 文件（覆盖前两轮同 abbr 的条目 + 注册全新因子）──
#   重写版（13个）：T_* 别名类，abbr 与主文件相同，transfer 版优先
#   纯新因子（26个）：主文件中不存在的 abbr
_register(
    # A1 重写
    T_RsFactor, T_TesFactor, T_NeiFactor,
    # A2 纯新
    BmjFactor, BmqFactor, SrFactor, SgFactor,
    # A3 重写 + 纯新
    T_IvcFactor,
    CeiFactor, CdiFactor, IvgFactor,
    OaFactor, TaFactor,
    DWcFactor, DCoaFactor, DColFactor, DNcoFactor, DncaFactor, DnclFactor,
    DfinFactor, DstiFactor, DltiFactor, DfnlFactor, DbeFactor,
    PoaFactor, PtaFactor, NxfFactor, NefFactor, NdfFactor,
    # A6 重写 + 纯新
    T_MeFactor, T_IvffFactor, T_TurFactor, T_DtvFactor, T_PpsFactor,
    T_AmiFactor, T_TsFactor, T_SrevFactor, T_ShlFactor,
    MdrFactor,
)

# ---------------------------------------------------------------------------
# 5.  工厂函数
# ---------------------------------------------------------------------------

def create(name: str, **kwargs: Any) -> BaseFactor:
    """
    按 abbr 名称实例化因子对象。

    Parameters
    ----------
    name : str
        因子 abbr（大小写不敏感），对应 REGISTRY 中的键。
    **kwargs
        传递给因子类构造函数的额外参数。

    Returns
    -------
    BaseFactor

    Raises
    ------
    ValueError
        如果 name 不在注册表中。
    """
    key = name.lower()
    if key not in REGISTRY:
        supported = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"Unknown factor name={name!r}.\n"
            f"Supported ({len(REGISTRY)} total): {supported}"
        )
    return REGISTRY[key](**kwargs)


# ---------------------------------------------------------------------------
# 6.  覆盖 BaseFactor.save() 使输出到 clean_output/
#     用猴子补丁方式，避免修改主文件
# ---------------------------------------------------------------------------

def _patched_save(self: BaseFactor, df, start_date=None, end_date=None):
    """将因子结果保存到 <root>/clean_output/ 而非原 output/ 目录。"""
    import pandas as pd

    if start_date is None or end_date is None:
        dates = df.index.get_level_values("ts")
        if start_date is None:
            start_date = (
                str(dates.min()) + "01"
                if pd.api.types.is_integer_dtype(dates)
                else dates.min().strftime("%Y%m%d")
            )
        if end_date is None:
            end_date = (
                str(dates.max()) + "28"
                if pd.api.types.is_integer_dtype(dates)
                else dates.max().strftime("%Y%m%d")
            )

    filename = f"{self.factor_id}_{self.abbr}_{start_date}-{end_date}.parquet"
    filepath = CLEAN_OUTPUT_DIR / filename
    df.to_parquet(filepath)
    print(f"  ✅ Saved → {filepath.relative_to(ROOT_DIR)}")
    return filepath


# 打补丁
BaseFactor.save = _patched_save  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# 7.  main() —— 遍历注册表，依次计算并保存
# ---------------------------------------------------------------------------

def main() -> None:
    """
    遍历 REGISTRY 中所有因子，依次：
      1. calculate()
      2. as_factor_frame()
      3. save() → clean_output/
    使用 tqdm 显示进度与耗时。
    """
    factor_names = sorted(REGISTRY.keys())
    print(f"\n{'='*60}")
    print(f"  因子工厂启动：共 {len(factor_names)} 个因子待计算")
    print(f"  输出目录：{CLEAN_OUTPUT_DIR}")
    print(f"{'='*60}\n")

    success, failed = [], []

    for name in tqdm(factor_names, desc="计算进度", unit="factor"):
        try:
            factor = create(name)
            tqdm.write(f"\n▶ [{factor.factor_id}] {factor.__class__.__name__}  (abbr={factor.abbr})")

            # 计算
            df_raw = factor.calculate()

            # 转换为 factor frame
            df_ff = factor.as_factor_frame(df_raw)

            # 保存到 clean_output/
            factor.save(df_ff)

            success.append(name)

        except Exception as exc:
            tqdm.write(f"  ❌ 失败：{name}  →  {exc}")
            tqdm.write(traceback.format_exc())
            failed.append((name, str(exc)))

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"  完成：{len(success)}/{len(factor_names)} 个因子成功")
    if failed:
        print(f"  失败因子（{len(failed)} 个）：")
        for nm, err in failed:
            print(f"    - {nm}: {err}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()