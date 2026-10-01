"""领域异常。

BuildError 表示「作业应当失败并记录原因」的情形（例如牛顿迭代不收敛）；
ValidationError 表示作业开始前的输入校验失败（HTTP 400，不产生作业）。
"""


class ScorecardError(Exception):
    """所有领域异常的基类。"""


class ValidationError(ScorecardError):
    """作业开始前的输入校验失败。"""


class BuildError(ScorecardError):
    """建卡过程中作业失败，原因需要落库可查。"""
