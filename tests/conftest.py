"""pytest 共享配置（健壮版 sys.path 注入）。

pytest 在导入本目录任何测试模块**之前**加载 conftest.py。这里把仓库根与 ``tools/wood``
提前加入 ``sys.path``，使 tests/ 下测试能 import 引擎模块（engine.*）与 tools/wood 里的
纯 stdlib 汇总工具（summarize_srff_v11_causal / _expert_capacity / summarize_srff_v12_pilot）。

为规避不同 pytest 版本 / 导入模式 / 符号链接下 ``__file__`` 解析差异，这里用**多候选根**
（本文件相对父目录、resolve 后的父目录、当前工作目录），且只注入**真实存在**的目录。
只要 pytest 从仓库根运行（验收手册即如此），CWD 候选必定命中 tools/wood。
"""

import sys
from pathlib import Path

_CANDIDATE_ROOTS = []
try:
    _here = Path(__file__).parent
    _CANDIDATE_ROOTS.append(_here.parent)                 # tests/ 的父目录（不 resolve）
    _CANDIDATE_ROOTS.append(Path(__file__).resolve().parents[1])  # resolve 版
except Exception:                                          # pragma: no cover
    pass
_CANDIDATE_ROOTS.append(Path.cwd())                        # 运行 pytest 的工作目录（通常为仓库根）

_added = set()
for _root in _CANDIDATE_ROOTS:
    for _p in (_root, _root / 'tools' / 'wood'):
        _ps = str(_p)
        try:
            _exists = _p.is_dir()
        except Exception:                                  # pragma: no cover
            _exists = False
        if _exists and _ps not in sys.path and _ps not in _added:
            sys.path.insert(0, _ps)
            _added.add(_ps)
